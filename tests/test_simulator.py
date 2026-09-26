import numpy as np
import pandas as pd
import pytest

from adaptive_exec.schedules import flat_curve
from adaptive_exec.simulator import (ExecSettings, MarketContext, MarketReplay, ParentOrder,
                                     run_parent_order)
from adaptive_exec.strategies import (TWAP, VWAP, AlmgrenChriss, HeuristicAdaptive, MLAdaptive,
                                      clamp_to_band)


@pytest.fixture(scope="module")
def replay(events):
    return MarketReplay(events)


@pytest.fixture(scope="module")
def market():
    return MarketContext(curve=flat_curve(), sigma_daily_bps=150.0, daily_volume=2_000_000)


ORDER = ParentOrder(side=+1, qty=3000, start=pd.Timestamp("2012-06-21 09:35"), horizon_s=600)


@pytest.mark.parametrize("strategy", [TWAP(), VWAP(), AlmgrenChriss(2.0), HeuristicAdaptive(0.5, 0.5)])
@pytest.mark.parametrize("side", [+1, -1])
def test_every_share_is_filled_or_charged(replay, market, strategy, side):
    o = ParentOrder(side=side, qty=ORDER.qty, start=ORDER.start, horizon_s=ORDER.horizon_s)
    r = run_parent_order(replay, o, strategy, market, ExecSettings())
    assert r["filled"] + r["unfilled"] == o.qty
    parts = sum(r[k] for k in ("spread_bps", "drift_bps", "impact_bps", "fees_bps", "opportunity_bps"))
    assert r["is_bps"] == pytest.approx(parts)
    assert 0 <= r["passive_share"] <= 1


def test_no_impact_setting_gives_zero_impact_cost(replay, market):
    r = run_parent_order(replay, ORDER, VWAP(), market, ExecSettings(impact_Y=0.0))
    assert r["impact_bps"] == 0.0
    r2 = run_parent_order(replay, ORDER, VWAP(), market, ExecSettings(impact_Y=1.0))
    assert r2["impact_bps"] > 0


def test_optimistic_fills_are_at_least_as_passive(replay, market):
    cons = run_parent_order(replay, ORDER, VWAP(), market, ExecSettings(fill_mode="conservative"))
    opt = run_parent_order(replay, ORDER, VWAP(), market, ExecSettings(fill_mode="optimistic"))
    assert opt["passive_share"] >= cons["passive_share"]


def test_passive_fills_never_use_trades_before_posting(events, market):
    """Delete every trade before the order starts: the result must not change."""
    t0 = ORDER.start
    before = events["is_trade"] & (events["ts"] <= t0)
    stripped = events.copy()
    stripped.loc[before, "is_trade"] = False
    a = run_parent_order(MarketReplay(events), ORDER, VWAP(), market, ExecSettings())
    b = run_parent_order(MarketReplay(stripped), ORDER, VWAP(), market, ExecSettings())
    assert a["is_bps"] == b["is_bps"] and a["passive_share"] == b["passive_share"]


def test_ml_strategy_crosses_on_a_strong_adverse_signal(replay, market):
    grid = pd.date_range("2012-06-21 09:30", "2012-06-21 09:50", freq="1s", unit="ns")
    ns = np.asarray(grid).view("int64")
    up = MarketContext(market.curve, market.sigma_daily_bps, market.daily_volume,
                       {"m": (ns, np.full(len(ns), 50.0))})       # price expected to rise 50 bps
    buy = run_parent_order(replay, ORDER, MLAdaptive("m", 0.5, 0.25), up, ExecSettings())
    assert buy["n_cross_decisions"] == ORDER.horizon_s // ORDER.slice_s
    assert buy["passive_share"] == 0.0
    sell = ParentOrder(side=-1, qty=ORDER.qty, start=ORDER.start, horizon_s=ORDER.horizon_s)
    s = run_parent_order(replay, sell, MLAdaptive("m", 0.5, 0.25), up, ExecSettings())
    assert s["n_post_decisions"] == sell.horizon_s // sell.slice_s


def test_clamp_to_band():
    assert clamp_to_band(900, 500, 1000, band=0.10) == 600
    assert clamp_to_band(100, 500, 1000, band=0.10) == 400
    assert clamp_to_band(550, 500, 1000, band=0.10) == 550


def test_fill_mid_is_the_book_before_the_whole_sweep():
    """A sweep logs several executions at one timestamp; the reference mid must be the
    book before the sweep, not the partially swept book between its messages."""
    t = pd.Timestamp("2012-06-21 09:30:01")
    rows = [  # ts, type, size, price, order_side, aggressor, is_trade, bid, bsz, ask, asz
        (t, 1, 100, 10.00, 1, 0, False, 10.00, 300, 10.01, 100),
        (t + pd.Timedelta(seconds=1), 4, 300, 10.00, 1, -1, True, 9.99, 500, 10.01, 100),
        (t + pd.Timedelta(seconds=1), 4, 200, 9.99, 1, -1, True, 9.99, 300, 10.01, 100),
    ]
    ev = pd.DataFrame(rows, columns=["ts", "type", "size", "price", "order_side", "aggressor",
                                     "is_trade", "bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1"])
    for i in range(2, 11):
        for c in ("bid_px", "bid_sz", "ask_px", "ask_sz"):
            ev[f"{c}_{i}"] = np.nan
    ev["mid"] = (ev["bid_px_1"] + ev["ask_px_1"]) / 2
    ev["spread"] = ev["ask_px_1"] - ev["bid_px_1"]
    ev["valid"] = True
    r = MarketReplay(ev)
    assert r.t_mid.tolist() == pytest.approx([10.005, 10.005])


def test_trace_accounts_for_every_share(replay, market):
    r = run_parent_order(replay, ORDER, HeuristicAdaptive(0.3, 0.5), market, ExecSettings(), trace=True)
    d = pd.DataFrame(r["decisions"])
    assert len(d) == ORDER.horizon_s // ORDER.slice_s
    assert d["filled_in_slice"].sum() + (r["filled"] - d["done_cum"].iloc[-1]) == r["filled"]
    assert sum(f.qty for f in r["fills"]) == r["filled"]
    passive = sum(f.qty for f in r["fills"] if f.passive)
    assert passive / ORDER.qty == pytest.approx(r["passive_share"])
    # resting orders sit at the touch that was displayed at the decision time
    posted = d.dropna(subset=["post_px"])
    assert (posted["post_px"] == posted["bid"]).all()
