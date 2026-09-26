"""Synthetic order book events in raw LOBSTER format, for tests and dry runs.

This is a level-based book on a one-tick grid, not an order-level one. Each event
adds to, cancels from or executes against one price level. The side of market
orders follows a slowly varying latent flow, so order flow has some weak,
deliberately planted predictive power. Results on this data say nothing about
real markets. They only show that the pipeline runs end to end.
"""
import numpy as np
import pandas as pd

from .loaders import book_columns


def generate_lobster_day(
    seed: int = 0,
    start_s: float = 34_200.0,
    end_s: float = 57_600.0,
    rate_per_s: float = 6.0,
    start_price: float = 30.0,
    tick: float = 0.01,
    levels: int = 10,
    flow_persistence_s: float = 60.0,
    flow_strength: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (message, orderbook) frames exactly as LOBSTER's CSVs would be read."""
    rng = np.random.default_rng(seed)
    unit = int(round(tick * 10_000))              # LOBSTER price units per tick
    bid = int(round(start_price / tick))          # prices in ticks
    ask = bid + 1
    qb = rng.integers(3, 20, levels) * 100
    qa = rng.integers(3, 20, levels) * 100

    msgs, books = [], []
    order_id = 1
    flow = 0.0
    t = start_s
    span = end_s - start_s

    def record(t_, typ, size, px_ticks, direction):
        msgs.append((t_, typ, order_id, int(size), px_ticks * unit, direction))
        row = np.empty(4 * levels, dtype=np.int64)
        row[0::4] = (ask + np.arange(levels)) * unit
        row[1::4] = qa
        row[2::4] = (bid - np.arange(levels)) * unit
        row[3::4] = qb
        books.append(row)

    while True:
        x = (t - start_s) / span
        intensity = rate_per_s * (0.6 + 1.6 * (2 * x - 1) ** 2)   # busy open and close
        dt = rng.exponential(1.0 / intensity)
        t += dt
        if t >= end_s:
            break
        flow += -flow * dt / flow_persistence_s + np.sqrt(2 * dt / flow_persistence_s) * rng.standard_normal()
        order_id += 1
        u = rng.random()

        if u < 0.50:                                   # new limit order
            side = 1 if rng.random() < 0.5 else -1
            size = rng.integers(1, 6) * 100
            if ask - bid > 1 and rng.random() < 0.6:   # improve the quote inside the spread
                if side == 1:
                    bid += 1
                    qb = np.concatenate([[size], qb[:-1]])
                    record(t, 1, size, bid, 1)
                else:
                    ask -= 1
                    qa = np.concatenate([[size], qa[:-1]])
                    record(t, 1, size, ask, -1)
                continue
            lvl = min(int(rng.geometric(0.35)) - 1, levels - 1)
            if side == 1:
                qb[lvl] += size
                record(t, 1, size, bid - lvl, 1)
            else:
                qa[lvl] += size
                record(t, 1, size, ask + lvl, -1)

        elif u < 0.88:                                 # cancellation
            side = 1 if rng.random() < 0.5 else -1
            q = qb if side == 1 else qa
            lvl = int(rng.choice(levels, p=q / q.sum()))
            want = int(rng.integers(1, 6) * 100)
            floor = 0 if lvl == 0 else 100             # deeper levels never empty
            size = min(want, int(q[lvl]) - floor)
            if size <= 0:
                continue
            q[lvl] -= size
            px = bid - lvl if side == 1 else ask + lvl
            typ = 3 if q[lvl] == 0 or size == want else 2
            if q[lvl] == 0:                            # best level emptied: book shifts
                if side == 1:
                    qb = np.concatenate([qb[1:], [rng.integers(3, 20) * 100]])
                    bid -= 1
                else:
                    qa = np.concatenate([qa[1:], [rng.integers(3, 20) * 100]])
                    ask += 1
            record(t, typ, size, px, side)

        else:                                          # marketable order walks the book
            p_buy = 0.5 + 0.5 * flow_strength * np.tanh(flow)
            aggressor = 1 if rng.random() < p_buy else -1
            left = int(rng.integers(1, 8) * 100)
            while left > 0:
                if aggressor == 1:
                    take, px = min(left, int(qa[0])), ask
                    qa[0] -= take
                    if qa[0] == 0:
                        qa = np.concatenate([qa[1:], [rng.integers(3, 20) * 100]])
                        ask += 1
                    record(t, 4, take, px, -1)         # direction = resting sell order
                else:
                    take, px = min(left, int(qb[0])), bid
                    qb[0] -= take
                    if qb[0] == 0:
                        qb = np.concatenate([qb[1:], [rng.integers(3, 20) * 100]])
                        bid -= 1
                    record(t, 4, take, px, 1)
                left -= take

    msg = pd.DataFrame(msgs, columns=["time", "type", "order_id", "size", "price", "direction"])
    book = pd.DataFrame(np.vstack(books), columns=book_columns(levels))
    return msg, book


def write_lobster_day(directory, symbol: str, date: str, levels: int = 10, **kwargs) -> None:
    """Write a synthetic day with LOBSTER's file names and header-less CSV layout."""
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    msg, book = generate_lobster_day(levels=levels, **kwargs)
    stem = f"{symbol}_{date}_34200000_57600000"
    msg.to_csv(d / f"{stem}_message_{levels}.csv", header=False, index=False)
    book.to_csv(d / f"{stem}_orderbook_{levels}.csv", header=False, index=False)
