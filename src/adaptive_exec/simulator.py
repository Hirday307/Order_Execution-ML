"""Parent-order replay: insert our child orders into the historical event stream.

Every `slice_s` seconds (decision time t_k):
  1. The previous resting child order is cancelled (its unfilled shares are dropped).
  2. The strategy picks a tactic and a speed multiplier from the state at t_k.
  3. If cumulative fills are more than `catchup_threshold` of the order behind
     schedule, the shortfall is crossed back to the schedule (catch-up).
  4. The slice quantity is clamped so cumulative shares stay within the band.
  5. Crossing walks the book snapshot at t_k (at most once per decision, so the
     book has a full slice to refresh). Posting rests at the best price, at the
     back of the queue, and is fed only trades with ts in (t_k, t_k+1].
At the horizon, the remainder is crossed; what the displayed book cannot absorb
is charged as opportunity cost at the final mid plus half the spread.
"""
from dataclasses import dataclass, field
import itertools

import numpy as np
import pandas as pd

from .features import NS, as_ns
from .fills import PassiveOrder, px_int, walk_book
from .impact import TransientImpact, calibrate_lambda
from .metrics import Fill, decompose
from .schedules import slice_weights, vwap
from .strategies import DecisionContext, Strategy, clamp_to_band


@dataclass(frozen=True)
class ParentOrder:
    side: int                 # +1 buy, -1 sell
    qty: int                  # shares
    start: pd.Timestamp
    horizon_s: int = 1800     # 30 minutes
    slice_s: int = 30         # one decision every 30 seconds
    size_target: float = float("nan")   # fraction of expected volume (for reporting)


@dataclass(frozen=True)
class ExecSettings:
    taker_fee: float = 0.0030           # $ per share
    maker_rebate: float = 0.0020        # $ per share
    band: float = 0.10
    catchup_threshold: float = 0.10
    fill_mode: str = "conservative"     # or "optimistic"
    impact_Y: float = 0.5
    impact_half_life_s: float = 300.0


@dataclass
class MarketContext:
    """Inputs that must come from outside the traded period (except where noted)."""
    curve: pd.Series                    # expected volume share per 5-min bucket (past days)
    sigma_daily_bps: float              # for impact calibration only
    daily_volume: float                 # for impact calibration and order sizing
    predictions: dict = field(default_factory=dict)   # model -> (grid_ns, pred_bps)


class MarketReplay:
    def __init__(self, events: pd.DataFrame, levels: int = 10):
        ns = as_ns(events["ts"])
        valid = events["valid"].to_numpy()
        self.levels = levels
        self.v_ns = ns[valid]
        v = events[valid]
        self.bid_px = v[[f"bid_px_{i}" for i in range(1, levels + 1)]].to_numpy()
        self.bid_sz = v[[f"bid_sz_{i}" for i in range(1, levels + 1)]].to_numpy()
        self.ask_px = v[[f"ask_px_{i}" for i in range(1, levels + 1)]].to_numpy()
        self.ask_sz = v[[f"ask_sz_{i}" for i in range(1, levels + 1)]].to_numpy()
        self.mid = (self.bid_px[:, 0] + self.ask_px[:, 0]) / 2

        pos = np.arange(len(events))
        tr = events["is_trade"].to_numpy()
        self.t_ns, self.t_pos = ns[tr], pos[tr]
        self.t_px = events["price"].to_numpy()[tr]
        self.t_pxi = np.rint(self.t_px * 10_000).astype(np.int64)
        self.t_sz = events["size"].to_numpy()[tr]
        self.t_agg = events["aggressor"].to_numpy()[tr]
        # Historical mid at a fill = last valid book with an EARLIER timestamp. A sweep
        # logs one execution per level at the same timestamp, and the book between
        # those messages already reflects the sweep.
        j = np.searchsorted(self.v_ns, self.t_ns, side="left") - 1
        self.t_mid = np.where(j >= 0, self.mid[np.maximum(j, 0)], np.nan)

        cx = events["type"].isin([2, 3]).to_numpy()
        self.c_ns, self.c_pos = ns[cx], pos[cx]
        self.c_pxi = np.rint(events["price"].to_numpy()[cx] * 10_000).astype(np.int64)
        self.c_px = events["price"].to_numpy()[cx]
        self.c_sz = events["size"].to_numpy()[cx]
        self.c_side = events["order_side"].to_numpy()[cx]

    def snapshot(self, t_ns: int) -> int:
        k = int(np.searchsorted(self.v_ns, t_ns, side="right") - 1)
        if k < 0:
            raise ValueError("no valid book at or before this time")
        return k

    def opposite_levels(self, k: int, side: int):
        px, sz = (self.ask_px[k], self.ask_sz[k]) if side == 1 else (self.bid_px[k], self.bid_sz[k])
        return list(zip(px, np.nan_to_num(sz)))

    def traded_volume(self, t0_ns: int, t1_ns: int) -> float:
        a, b = np.searchsorted(self.t_ns, [t0_ns, t1_ns], side="right")
        return float(self.t_sz[a:b].sum())


def _lookup(pred, t_ns):
    if pred is None:
        return np.nan
    grid_ns, vals = pred
    i = np.searchsorted(grid_ns, t_ns, side="right") - 1
    return float(vals[i]) if i >= 0 else np.nan


def order_weights(curve: pd.Series, order: ParentOrder) -> np.ndarray:
    """Expected volume share of each slice of the order (uniform if the curve is empty)."""
    n = order.horizon_s // order.slice_s
    starts = pd.date_range(order.start, periods=n, freq=f"{order.slice_s}s")
    w = slice_weights(curve, starts, order.slice_s)
    return w if w.sum() > 0 else np.ones(n)


def run_parent_order(replay: MarketReplay, order: ParentOrder, strategy: Strategy,
                     market: MarketContext, settings: ExecSettings,
                     weights: np.ndarray | None = None, trace: bool = False) -> dict:
    """Execute one parent order. `weights` (from `order_weights`) can be passed in to
    avoid recomputing them for every strategy. With `trace=True` the result also
    holds every decision and fill, for checking an order by hand."""
    side, Q = order.side, int(order.qty)
    S = order.slice_s * NS
    n = order.horizon_s // order.slice_s
    t0 = int(as_ns(pd.DatetimeIndex([order.start]))[0])
    starts = t0 + np.arange(n, dtype=np.int64) * S

    if weights is None:
        weights = order_weights(market.curve, order)
    sched = strategy.schedule(Q, weights)
    sched_cum = np.cumsum(sched)
    # Impact strength is calibrated on the VWAP schedule, identically for every strategy
    lam = calibrate_lambda(settings.impact_Y, market.sigma_daily_bps, Q, market.daily_volume,
                           vwap(Q, weights), order.slice_s, settings.impact_half_life_s)
    impact = TransientImpact(lam, settings.impact_half_life_s)
    pred = market.predictions.get(strategy.model) if strategy.model else None

    k0 = replay.snapshot(t0)
    A = replay.mid[k0]
    fills: list[Fill] = []
    decisions: list[dict] = []
    done, passive_done, n_catchups = 0, 0, 0
    tactic_counts = {"baseline": 0, "post": 0, "cross": 0}
    posted = {}                       # details of the latest resting order, for the trace

    def secs(t_ns):
        return (t_ns - t0) / NS

    def cross(t_ns, k, qty):
        got_fills, _ = walk_book(qty, replay.opposite_levels(k, side))
        push = impact.current_bps(secs(t_ns))
        total = 0
        for px, q in got_fills:
            fills.append(Fill(secs(t_ns), px, px * (1 + side * push / 1e4), replay.mid[k], q,
                              False, settings.taker_fee * q))
            total += q
        if total:
            impact.add_fill(secs(t_ns), total)
        return total

    def post(t_start, t_stop, k, qty):
        if side == 1:
            px, ahead = replay.bid_px[k, 0], replay.bid_sz[k, 0]
        else:
            px, ahead = replay.ask_px[k, 0], replay.ask_sz[k, 0]
        order_ = PassiveOrder(side, px, qty, ahead)
        posted.update(post_px=px, queue_ahead=ahead)
        pi = px_int(px)
        a, b = np.searchsorted(replay.t_ns, [t_start, t_stop], side="right")
        tpxi, tagg = replay.t_pxi[a:b], replay.t_agg[a:b]
        through = tpxi < pi if side == 1 else tpxi > pi
        hits = a + np.flatnonzero((tagg == -side) & ((tpxi == pi) | through))
        stream = [(replay.t_pos[i], 0, i) for i in hits]
        if settings.fill_mode == "optimistic":
            ca, cb = np.searchsorted(replay.c_ns, [t_start, t_stop], side="right")
            ours = ca + np.flatnonzero((replay.c_side[ca:cb] == side) & (replay.c_pxi[ca:cb] == pi))
            stream += [(replay.c_pos[i], 1, i) for i in ours]
            stream.sort()
        for _, kind, i in stream:
            if kind == 1:
                order_.on_cancel(replay.c_px[i], replay.c_sz[i], replay.c_side[i])
                continue
            q = order_.on_trade(replay.t_px[i], replay.t_sz[i], replay.t_agg[i])
            if q:
                t = secs(replay.t_ns[i])
                m = replay.t_mid[i] if np.isfinite(replay.t_mid[i]) else replay.mid[k]
                push = impact.current_bps(t)
                fills.append(Fill(t, px, px * (1 + side * push / 1e4), m, q, True,
                                  -settings.maker_rebate * q))
                impact.add_fill(t, q)
            if order_.remaining == 0:
                break
        return order_.filled

    for i in range(n):
        tk = int(starts[i])
        k = replay.snapshot(tk)
        mid = replay.mid[k]
        bsz, asz = replay.bid_sz[k, 0], replay.ask_sz[k, 0]
        ctx = DecisionContext(
            side=side, pred_bps=_lookup(pred, tk),
            imb_l1=(bsz - asz) / (bsz + asz),
            spread_bps=(replay.ask_px[k, 0] - replay.bid_px[k, 0]) / mid * 1e4,
            taker_bps=settings.taker_fee / mid * 1e4,
            rebate_bps=settings.maker_rebate / mid * 1e4,
        )
        tactic, mult = strategy.decide(ctx)
        tactic_counts[tactic] += 1
        done_before, fills_before = done, len(fills)
        posted.clear()

        behind = (sched_cum[i - 1] if i else 0) - done
        catch = int(behind) if behind > settings.catchup_threshold * Q else 0
        planned = clamp_to_band(done + catch + sched[i] * mult, sched_cum[i], Q, settings.band)
        slice_qty = max(0, planned - done - catch)
        to_cross = catch + (slice_qty if tactic == "cross" else 0)
        to_post = 0 if tactic == "cross" else slice_qty

        if to_cross > 0:
            done += cross(tk, k, min(to_cross, Q - done))
            n_catchups += catch > 0
        to_post = min(to_post, Q - done)
        if to_post > 0:
            got = post(tk, tk + S, k, to_post)
            done += got
            passive_done += got
        if trace:
            decisions.append({
                "slice": i, "t_s": secs(tk), "bid": replay.bid_px[k, 0], "bid_size": bsz,
                "ask": replay.ask_px[k, 0], "ask_size": asz, "pred_bps": ctx.pred_bps,
                "tactic": tactic, "speed": mult, "scheduled": int(sched[i]),
                "schedule_cum": int(sched_cum[i]), "catch_up": catch, "to_cross": int(to_cross),
                "to_post": int(to_post), "post_px": posted.get("post_px", np.nan),
                "queue_ahead": posted.get("queue_ahead", np.nan),
                "filled_in_slice": done - done_before, "n_fills": len(fills) - fills_before,
                "done_cum": done,
            })

    t_end = t0 + n * S
    k_end = replay.snapshot(t_end)
    left = Q - done
    unfilled = 0
    if left > 0:
        got = cross(t_end, k_end, left)
        done += got
        unfilled = left - got
    final_half_spread = (replay.ask_px[k_end, 0] - replay.bid_px[k_end, 0]) / 2
    cost = decompose(side, A, Q, fills, unfilled, replay.mid[k_end], final_half_spread)
    mkt_vol = replay.traded_volume(t0, t_end)
    return {
        "strategy": strategy.name, "params": strategy.params(),
        "side": side, "qty": Q, "start": order.start, "size_target": order.size_target,
        **cost,
        "filled": done, "unfilled": unfilled,
        "passive_share": passive_done / Q,
        "n_catchups": n_catchups,
        "n_cross_decisions": tactic_counts["cross"], "n_post_decisions": tactic_counts["post"],
        "arrival_mid": A, "market_volume": mkt_vol,
        "participation": done / mkt_vol if mkt_vol > 0 else np.nan,
        "impact_lambda": lam,
        **({"decisions": decisions, "fills": fills, "end_mid": replay.mid[k_end],
            "end_half_spread": final_half_spread} if trace else {}),
    }


def expected_volume(curve: pd.Series, reference_daily_volume: float, start: pd.Timestamp,
                    horizon_min: int, slice_s: int = 30) -> float:
    starts = pd.date_range(start, periods=horizon_min * 60 // slice_s, freq=f"{slice_s}s")
    return reference_daily_volume * slice_weights(curve, starts, slice_s).sum()


def make_parent_orders(window_start, window_end, expected_volume_fn,
                       targets=(0.02, 0.05, 0.10), every_min=5, horizon_min=30,
                       slice_s=30) -> list[ParentOrder]:
    last_start = pd.Timestamp(window_end) - pd.Timedelta(minutes=horizon_min)
    starts = pd.date_range(window_start, last_start, freq=f"{every_min}min")
    orders = []
    for start, p, side in itertools.product(starts, targets, (+1, -1)):
        qty = int(round(p * expected_volume_fn(start, horizon_min)))
        if qty > 0:
            orders.append(ParentOrder(side=side, qty=qty, start=start, horizon_s=horizon_min * 60,
                                      slice_s=slice_s, size_target=p))
    return orders
