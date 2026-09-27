"""LOBSTER and Databento data -> one internal event format.

Internal format: one row per event, in time order, with the book state *after*
the event.

    ts            datetime64[ns], New York local time, tz-naive
    type          LOBSTER event code: 1 add, 2 partial cancel, 3 delete,
                  4 visible execution, 5 hidden execution, 6 cross; 0 other;
                  8 book update that removes a traded quantity (Databento only;
                  not a cancellation, so the queue model ignores it)
    size, price   event size (shares) and price (dollars)
    order_side    +1 / -1 side of the order the event refers to (for executions,
                  the resting order); 0 if unknown
    aggressor     for trades: +1 buyer-initiated, -1 seller-initiated; else 0
    is_trade      True for continuous-trading executions (types 4 and 5)
    ask_px_i, ask_sz_i, bid_px_i, bid_sz_i   for i = 1..levels (NaN if empty)
    mid, spread
    valid         book is two-sided and not locked or crossed
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"


def book_columns(levels: int) -> list[str]:
    return [f"{side}_{kind}_{i}" for i in range(1, levels + 1)
            for side, kind in (("ask", "px"), ("ask", "sz"), ("bid", "px"), ("bid", "sz"))]


def classify_hidden_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Quote rule for hidden executions (type 5): above the mid buyer-initiated,
    below seller-initiated, at the mid unknown (0). The feed's side flag for these
    trades is not informative (Nasdaq ITCH 5.0 sets it to 'B' on every one).
    Hidden orders are not in the displayed book, so the row's book is the one
    the trade met."""
    hidden = df["type"].to_numpy() == 5
    if hidden.any():
        mid = (df["bid_px_1"].to_numpy() + df["ask_px_1"].to_numpy()) / 2
        sign = np.nan_to_num(np.sign(np.round(df["price"].to_numpy() - mid, 6))).astype(int)
        df.loc[hidden, "aggressor"] = sign[hidden]
        df.loc[hidden, "order_side"] = -sign[hidden]
    return df


def _finish(df: pd.DataFrame, levels: int) -> pd.DataFrame:
    df["mid"] = (df["ask_px_1"] + df["bid_px_1"]) / 2
    df["spread"] = df["ask_px_1"] - df["bid_px_1"]
    df["valid"] = (df["spread"] > 0).fillna(False)
    cols = (["ts", "type", "size", "price", "order_side", "aggressor", "is_trade"]
            + book_columns(levels) + ["mid", "spread", "valid"])
    return df[cols].reset_index(drop=True)


# ----------------------------------------------------------------------- LOBSTER

def lobster_to_events(msg: pd.DataFrame, book: pd.DataFrame, date: str,
                      levels: int = 10) -> pd.DataFrame:
    if len(msg) != len(book):
        raise ValueError("message and orderbook have different row counts")
    book = book.copy()
    book.columns = book_columns(levels)
    for i in range(1, levels + 1):
        book[f"ask_px_{i}"] = book[f"ask_px_{i}"] / 10_000
        book[f"bid_px_{i}"] = book[f"bid_px_{i}"] / 10_000
        # Empty levels hold dummy prices: very large for asks, negative for bids
        empty_ask = book[f"ask_px_{i}"] > 100_000
        empty_bid = book[f"bid_px_{i}"] < 0
        book.loc[empty_ask, [f"ask_px_{i}", f"ask_sz_{i}"]] = np.nan
        book.loc[empty_bid, [f"bid_px_{i}", f"bid_sz_{i}"]] = np.nan

    df = pd.concat([msg.reset_index(drop=True), book.reset_index(drop=True)], axis=1)
    df["ts"] = (pd.Timestamp(date) + pd.to_timedelta(df["time"], unit="s")).astype("datetime64[ns]")
    df = df[df["type"] != 7].copy()                 # trading-halt markers
    df["price"] = df["price"] / 10_000
    df["is_trade"] = df["type"].isin([4, 5])
    # LOBSTER's direction is the side of the RESTING order that was hit
    df["aggressor"] = np.where(df["is_trade"], -df["direction"], 0).astype(int)
    df["order_side"] = df["direction"].astype(int)
    return _finish(classify_hidden_trades(df), levels)


def load_lobster(msg_path, book_path, date: str, levels: int = 10) -> pd.DataFrame:
    msg = pd.read_csv(msg_path, header=None, usecols=range(6),
                      names=["time", "type", "order_id", "size", "price", "direction"])
    book = pd.read_csv(book_path, header=None)
    return lobster_to_events(msg, book, date, levels)


def find_lobster_files(raw_dir, symbol: str, levels: int = 10) -> list[tuple[str, Path, Path]]:
    """[(date, message_path, orderbook_path), ...] for one symbol, sorted by date."""
    out = []
    for m in sorted(Path(raw_dir).rglob(f"{symbol}_*_message_{levels}.csv")):
        date = re.match(rf"{symbol}_(\d{{4}}-\d{{2}}-\d{{2}})_", m.name).group(1)
        out.append((date, m, m.with_name(m.name.replace("_message_", "_orderbook_"))))
    return out


# --------------------------------------------------------------------- Databento

_DB_ACTION_TO_TYPE = {"A": 1, "C": 3, "M": 0, "T": 4, "F": 0, "R": 0, "N": 0}
_DB_SIDE = {"B": 1, "A": -1, "N": 0}
FILL_REMOVAL = 8


def _utc_ns(values) -> np.ndarray:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]").view("int64")


def mark_fill_removals(types, is_trade, ts_event_ns, price, order_side) -> np.ndarray:
    """Re-label cancels that only remove a traded quantity from the book.

    In MBP data a trade is followed by a book update that takes the filled size
    out of the level. When that update is a cancel at the trade's price, on the
    resting side, with the same exchange timestamp, it is not a real cancellation.
    """
    types = np.asarray(types).copy()
    pxi = np.where(np.isfinite(price), np.rint(np.nan_to_num(price) * 10_000), -1).astype(np.int64)
    cancel = types == 3
    trades = pd.DataFrame({"t": ts_event_ns[is_trade], "p": pxi[is_trade],
                           "s": order_side[is_trade]}).drop_duplicates()
    cancels = pd.DataFrame({"i": np.flatnonzero(cancel), "t": ts_event_ns[cancel],
                            "p": pxi[cancel], "s": order_side[cancel]})
    hit = cancels.merge(trades, on=["t", "p", "s"], how="inner")["i"].to_numpy()
    types[hit] = FILL_REMOVAL
    return types


def databento_to_events(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """Convert an `mbp-10` DBNStore.to_df() frame (one symbol) to the internal format.

    Databento's `side` on a trade is the AGGRESSOR ('B' buyer-initiated, 'A'
    seller-initiated), unlike LOBSTER. Check this against Databento's schema docs.
    """
    df = df.copy()
    recv = df.index if isinstance(df.index, pd.DatetimeIndex) else df["ts_recv"]
    ts = pd.DatetimeIndex(recv)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    out = pd.DataFrame({"ts": ts.tz_convert("America/New_York").tz_localize(None).astype("datetime64[ns]")})
    price = df["price"].to_numpy(dtype=float)
    if np.issubdtype(df["price"].dtype, np.integer):   # fixed-point prices (1e-9 dollars)
        price = price / 1e9
    action = df["action"].astype(str).to_numpy()
    side = df["side"].astype(str).map(_DB_SIDE).fillna(0).astype(int).to_numpy()
    is_trade = action == "T"
    order_side = np.where(is_trade, -side, side)
    types = pd.Series(action).map(_DB_ACTION_TO_TYPE).fillna(0).astype(int).to_numpy()
    ts_event = _utc_ns(df["ts_event"]) if "ts_event" in df else _utc_ns(recv)
    out["type"] = mark_fill_removals(types, is_trade, ts_event, price, order_side)
    out["size"] = df["size"].to_numpy()
    out["price"] = price
    out["is_trade"] = is_trade
    out["aggressor"] = np.where(is_trade, side, 0)
    out["order_side"] = order_side
    for i in range(1, levels + 1):
        j = f"{i - 1:02d}"
        for s, db in (("ask", "ask"), ("bid", "bid")):
            px = df[f"{db}_px_{j}"].to_numpy(dtype=float)
            if np.issubdtype(df[f"{db}_px_{j}"].dtype, np.integer):
                px = px / 1e9
            sz = df[f"{db}_sz_{j}"].to_numpy(dtype=float)
            empty = ~np.isfinite(px) | (sz <= 0) | (px <= 0) | (px > 1e6)
            out[f"{s}_px_{i}"] = np.where(empty, np.nan, px)
            out[f"{s}_sz_{i}"] = np.where(empty, np.nan, sz)
    tod = out["ts"] - out["ts"].dt.normalize()
    out = out[(tod >= pd.Timedelta(SESSION_OPEN + ":00")) & (tod < pd.Timedelta(SESSION_CLOSE + ":00"))]
    return _finish(out, levels)


def load_databento(path, levels: int = 10) -> pd.DataFrame:
    import databento as db
    return databento_to_events(db.DBNStore.from_file(path).to_df(), levels)
