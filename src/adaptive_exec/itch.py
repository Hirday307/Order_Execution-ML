"""Nasdaq TotalView-ITCH 5.0 sample files -> the internal event format.

Nasdaq publishes full-day ITCH 5.0 sample files (the feed LOBSTER is built from).
A day is about 12 GB uncompressed, so the work is done in two passes:

1. `extract_messages` streams the gzipped file once and keeps only the order
   messages of the chosen symbols (a few percent of the file).
2. `itch_to_events` rebuilds one symbol's book order by order and emits, for
   regular hours, one event per message that changes the top `levels` price
   levels or prints a trade, with the book after it.

File framing ("BinaryFILE"): each message is preceded by a 2-byte big-endian
length. Integers are big-endian, prices have 4 implied decimals, timestamps
are nanoseconds after midnight, Eastern time.

Event types in the output follow the LOBSTER codes used everywhere else:
1 add, 2 partial cancel, 3 delete (a replace is a delete then an add),
4 visible execution, 5 hidden execution ('P'), 6 cross or execution that is
not printed separately. `order_side` is the side of the order the event refers
to; for trades `aggressor` = -order_side.

Hidden executions: the 'P' message's buy/sell flag is 'B' on every message in
Nasdaq's sample files, so it says nothing about who initiated the trade. The
aggressor of a hidden trade is set by the quote rule instead (above the mid:
buyer-initiated; below: seller-initiated; at the mid: unknown, 0).
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .loaders import _finish, book_columns, classify_hidden_trades

T_R, T_A, T_F, T_E, T_C, T_X, T_D, T_U, T_P, T_Q = (ord(c) for c in "RAFECXDUPQ")
FIELDS = ("type", "ts", "ref", "ref2", "side", "shares", "price", "flag")


# ----------------------------------------------------------------- pass 1: scan

@njit(inline="always")
def _u16(b, i):
    return (np.int64(b[i]) << 8) | np.int64(b[i + 1])


@njit(inline="always")
def _u32(b, i):
    return (np.int64(b[i]) << 24) | (np.int64(b[i + 1]) << 16) | (np.int64(b[i + 2]) << 8) | np.int64(b[i + 3])


@njit(inline="always")
def _u48(b, i):
    return (_u16(b, i) << 32) | _u32(b, i + 2)


@njit(inline="always")
def _u64(b, i):
    return (_u32(b, i) << 32) | _u32(b, i + 4)


@njit(cache=True)
def _scan(buf, n, target, symtab, o_sym, o_type, o_ts, o_ref, o_ref2, o_side, o_shares, o_price, o_flag):
    """Parse the complete messages in buf[:n]; keep those of target symbols.
    Returns (bytes consumed, rows written)."""
    pos, k = 0, 0
    n_sym = symtab.shape[0]
    while pos + 2 <= n:
        length = _u16(buf, pos)
        if pos + 2 + length > n:
            break
        m = pos + 2
        t = buf[m]
        loc = _u16(buf, m + 1)
        if t == T_R:                                   # stock directory: locate -> symbol
            for s in range(n_sym):
                same = True
                for j in range(8):
                    if buf[m + 11 + j] != symtab[s, j]:
                        same = False
                        break
                if same:
                    target[loc] = s
        elif target[loc] >= 0 and (t == T_A or t == T_F or t == T_E or t == T_C or t == T_X
                                   or t == T_D or t == T_U or t == T_P or t == T_Q):
            o_sym[k] = target[loc]
            o_type[k] = t
            o_ts[k] = _u48(buf, m + 5)
            o_ref[k] = 0
            o_ref2[k] = 0
            o_side[k] = 0
            o_shares[k] = 0
            o_price[k] = 0
            o_flag[k] = 0
            if t == T_A or t == T_F:
                o_ref[k] = _u64(buf, m + 11)
                o_side[k] = 1 if buf[m + 19] == 66 else -1          # 'B'
                o_shares[k] = _u32(buf, m + 20)
                o_price[k] = _u32(buf, m + 32)
            elif t == T_E:
                o_ref[k] = _u64(buf, m + 11)
                o_shares[k] = _u32(buf, m + 19)
            elif t == T_C:
                o_ref[k] = _u64(buf, m + 11)
                o_shares[k] = _u32(buf, m + 19)
                o_flag[k] = buf[m + 31]                              # printable 'Y' / 'N'
                o_price[k] = _u32(buf, m + 32)
            elif t == T_X:
                o_ref[k] = _u64(buf, m + 11)
                o_shares[k] = _u32(buf, m + 19)
            elif t == T_D:
                o_ref[k] = _u64(buf, m + 11)
            elif t == T_U:
                o_ref[k] = _u64(buf, m + 11)
                o_ref2[k] = _u64(buf, m + 19)
                o_shares[k] = _u32(buf, m + 27)
                o_price[k] = _u32(buf, m + 31)
            elif t == T_P:
                o_side[k] = 1 if buf[m + 19] == 66 else -1
                o_shares[k] = _u32(buf, m + 20)
                o_price[k] = _u32(buf, m + 32)
            else:                                                    # 'Q' cross trade
                o_shares[k] = _u64(buf, m + 11)
                o_price[k] = _u32(buf, m + 27)
                o_flag[k] = buf[m + 39]                              # cross type
            k += 1
        pos += 2 + length
    return pos, k


def itch_date(path) -> str:
    """'03272019.NASDAQ_ITCH50.gz' -> '2019-03-27'."""
    s = Path(path).name[:8]
    return f"{s[4:8]}-{s[0:2]}-{s[2:4]}"


def _buffers(cap: int) -> list:
    return [np.zeros(cap, np.int16), np.zeros(cap, np.uint8)] + \
           [np.zeros(cap, np.int64) for _ in range(3)] + \
           [np.zeros(cap, np.int8), np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap, np.uint8)]


def extract_messages(gz_path, symbols, out_dir, chunk_bytes=64 << 20, read_bytes=8 << 20,
                     log=print) -> dict:
    """Stream a gzipped ITCH 5.0 file once; save each symbol's order messages to
    `{out_dir}/{symbol}.npz`. Returns {symbol: message count}."""
    symtab = np.array([list(s.ljust(8).encode()) for s in symbols], dtype=np.uint8)
    target = np.full(65536, -1, dtype=np.int16)
    outs, cap = _buffers(1), 1
    parts = []
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = bytearray()
    read, done = 0, False
    with open(gz_path, "rb") as fh:
        while not done:
            comp = fh.read(read_bytes)
            if comp:
                buf += dec.decompress(comp)
                read += len(comp)
            else:
                buf += dec.flush()
                done = True
            if len(buf) >= chunk_bytes or done:
                arr = np.frombuffer(buf, dtype=np.uint8)
                need = len(arr) // 14 + 1                 # the shortest message is 12 + 2 bytes
                if need > cap:                            # _scan does not bounds-check its writes
                    outs, cap = _buffers(need), need
                used, k = _scan(arr, len(arr), target, symtab, *outs)
                parts.append([o[:k].copy() for o in outs])
                del arr
                buf = buf[used:]
                if log and len(parts) % 20 == 0:
                    log(f"[itch] {read / 2**30:.1f} GiB compressed read, "
                        f"{sum(len(p[0]) for p in parts):,} messages kept")
    if len(buf):
        raise ValueError(f"{len(buf)} trailing bytes do not form a whole message")
    cols = [np.concatenate([p[i] for p in parts]) for i in range(len(outs))]
    sym = cols[0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for s_idx, s in enumerate(symbols):
        mask = sym == s_idx
        np.savez(out / f"{s}.npz", **{f: c[mask] for f, c in zip(FIELDS, cols[1:])})
        counts[s] = int(mask.sum())
    return counts


# ---------------------------------------------------------- pass 2: rebuild book

@njit(inline="always")
def _down(book, j):
    while j >= 0 and book[j] <= 0:
        j -= 1
    return j


@njit(inline="always")
def _up(book, j, n):
    while j < n and book[j] <= 0:
        j += 1
    return j


@njit(cache=True)
def _top(bid, ask, best_b, best_a, n, levels, max_scan, bpx, bsz, apx, asz):
    c, j, steps = 0, best_b, 0
    while c < levels and j >= 0 and steps < max_scan:
        if bid[j] > 0:
            bpx[c] = j
            bsz[c] = bid[j]
            c += 1
        j -= 1
        steps += 1
    for r in range(c, levels):
        bpx[r] = -1
        bsz[r] = 0
    c, j, steps = 0, best_a, 0
    while c < levels and j < n and steps < max_scan:
        if ask[j] > 0:
            apx[c] = j
            asz[c] = ask[j]
            c += 1
        j += 1
        steps += 1
    for r in range(c, levels):
        apx[r] = -1
        asz[r] = 0


@njit(cache=True)
def _rebuild(types, ts, shares, price, side, flag, own, tgt, n_orders, lo_tick, n_ticks,
             open_ns, close_ns, levels, max_scan,
             e_ts, e_type, e_size, e_price, e_side, e_trade, e_bpx, e_bsz, e_apx, e_asz):
    o_side = np.zeros(n_orders, np.int8)
    o_tick = np.full(n_orders, -1, np.int64)       # -1: outside the price grid (never in the top levels)
    o_px = np.zeros(n_orders, np.int64)            # limit price, 1e-4 dollars
    o_rem = np.zeros(n_orders, np.int64)
    bid = np.zeros(n_ticks, np.int64)
    ask = np.zeros(n_ticks, np.int64)
    best_b, best_a = -1, n_ticks
    cur = np.zeros((4, levels), np.int64)
    prev = np.full((4, levels), -2, np.int64)
    k = 0

    for i in range(types.shape[0]):
        t = types[i]
        # up to two book changes per message (a replace deletes then adds)
        for step in range(2):
            ev_type, ev_size, ev_px, ev_side, trade, touched = 0, 0, 0, 0, False, False
            if step == 0:
                if t == T_A or t == T_F:
                    o = own[i]
                    o_side[o], o_px[o], o_rem[o] = side[i], price[i], shares[i]
                    tick = price[i] // 100 - lo_tick
                    if price[i] % 100 == 0 and 0 <= tick < n_ticks:
                        o_tick[o] = tick
                        if side[i] > 0:
                            bid[tick] += shares[i]
                            if tick > best_b:
                                best_b = tick
                        else:
                            ask[tick] += shares[i]
                            if tick < best_a:
                                best_a = tick
                    ev_type, ev_size, ev_px, ev_side, touched = 1, shares[i], price[i], side[i], True
                elif t == T_E or t == T_C or t == T_X or t == T_D or t == T_U:
                    o = tgt[i]
                    if o < 0:
                        break
                    q = o_rem[o] if (t == T_D or t == T_U) else min(shares[i], o_rem[o])
                    o_rem[o] -= q
                    tick = o_tick[o]
                    if tick >= 0 and q > 0:
                        if o_side[o] > 0:
                            bid[tick] -= q
                            if tick == best_b and bid[tick] <= 0:
                                best_b = _down(bid, tick - 1)
                        else:
                            ask[tick] -= q
                            if tick == best_a and ask[tick] <= 0:
                                best_a = _up(ask, tick + 1, n_ticks)
                    ev_size, ev_side, ev_px, touched = q, o_side[o], o_px[o], True
                    if t == T_E:
                        ev_type, trade = 4, True
                    elif t == T_C:
                        ev_px = price[i]
                        if flag[i] == 89:                  # 'Y': printed as its own trade
                            ev_type, trade = 4, True
                        else:
                            ev_type = 6
                    elif t == T_X:
                        ev_type = 2
                    else:
                        ev_type = 3
                elif t == T_P:
                    ev_type, ev_size, ev_px, ev_side, trade, touched = 5, shares[i], price[i], side[i], True, True
                elif t == T_Q:
                    ev_type, ev_size, ev_px, touched = 6, shares[i], price[i], True
            else:
                if t != T_U or tgt[i] < 0:
                    break
                o = own[i]                                  # the replacement order
                o_side[o] = o_side[tgt[i]]
                o_px[o], o_rem[o] = price[i], shares[i]
                tick = price[i] // 100 - lo_tick
                if price[i] % 100 == 0 and 0 <= tick < n_ticks:
                    o_tick[o] = tick
                    if o_side[o] > 0:
                        bid[tick] += shares[i]
                        if tick > best_b:
                            best_b = tick
                    else:
                        ask[tick] += shares[i]
                        if tick < best_a:
                            best_a = tick
                ev_type, ev_size, ev_px, ev_side, touched = 1, shares[i], price[i], o_side[o], True
            if not touched or ts[i] < open_ns or ts[i] >= close_ns:
                continue
            _top(bid, ask, best_b, best_a, n_ticks, levels, max_scan, cur[0], cur[1], cur[2], cur[3])
            changed = False
            for a in range(4):
                for b in range(levels):
                    if cur[a, b] != prev[a, b]:
                        changed = True
            if changed or trade or ev_type == 6:
                e_ts[k], e_type[k], e_size[k], e_price[k] = ts[i], ev_type, ev_size, ev_px
                e_side[k], e_trade[k] = ev_side, trade
                for b in range(levels):
                    e_bpx[k, b], e_bsz[k, b] = cur[0, b], cur[1, b]
                    e_apx[k, b], e_asz[k, b] = cur[2, b], cur[3, b]
                k += 1
                for a in range(4):
                    for b in range(levels):
                        prev[a, b] = cur[a, b]
    return k


def itch_to_events(npz_path, date: str, levels: int = 10, grid_band: float = 0.4,
                   max_scan_ticks: int = 20_000) -> pd.DataFrame:
    """Rebuild one symbol's book from its extracted messages -> internal events.

    Displayed prices are on a one-cent grid spanning ±`grid_band` around the day's
    median add price; orders outside it (e.g. stub quotes) are tracked but can
    never reach the top levels of a liquid stock.
    """
    m = dict(np.load(npz_path))
    types, ts = m["type"], m["ts"]
    is_add, is_u = (types == T_A) | (types == T_F), types == T_U
    created = np.unique(np.concatenate([m["ref"][is_add], m["ref2"][is_u]]))
    own = np.full(len(types), -1, np.int64)
    own[is_add] = np.searchsorted(created, m["ref"][is_add])
    own[is_u] = np.searchsorted(created, m["ref2"][is_u])
    refers = np.isin(types, [T_E, T_C, T_X, T_D, T_U])
    idx = np.searchsorted(created, m["ref"][refers])
    ok = (idx < len(created)) & (created[np.minimum(idx, len(created) - 1)] == m["ref"][refers])
    tgt = np.full(len(types), -1, np.int64)
    tgt[np.flatnonzero(refers)[ok]] = idx[ok]

    add_px = m["price"][is_add]
    med = float(np.median(add_px[add_px > 0])) if (add_px > 0).any() else 1e6
    lo_tick = int(med * (1 - grid_band) // 100)
    n_ticks = int(med * (1 + grid_band) // 100) - lo_tick + 1
    open_ns, close_ns = int(9.5 * 3600e9), int(16 * 3600e9)

    cap = len(types) + int(is_u.sum()) + 1
    e_ts, e_type = np.zeros(cap, np.int64), np.zeros(cap, np.int8)
    e_size, e_price = np.zeros(cap, np.int64), np.zeros(cap, np.int64)
    e_side, e_trade = np.zeros(cap, np.int8), np.zeros(cap, np.bool_)
    e_bpx, e_bsz, e_apx, e_asz = (np.zeros((cap, levels), np.int32) for _ in range(4))
    k = _rebuild(types, ts, m["shares"], m["price"], m["side"], m["flag"], own, tgt, len(created),
                 lo_tick, n_ticks, open_ns, close_ns, levels, max_scan_ticks,
                 e_ts, e_type, e_size, e_price, e_side, e_trade, e_bpx, e_bsz, e_apx, e_asz)

    df = pd.DataFrame({
        "ts": (pd.Timestamp(date) + pd.to_timedelta(e_ts[:k], unit="ns")).astype("datetime64[ns]"),
        "type": e_type[:k].astype(int), "size": e_size[:k], "price": e_price[:k] / 10_000,
        "order_side": e_side[:k].astype(int),
    })
    df["is_trade"] = e_trade[:k]
    df["aggressor"] = np.where(df["is_trade"], -df["order_side"], 0)
    cols = {}
    for side, px, sz in (("bid", e_bpx, e_bsz), ("ask", e_apx, e_asz)):
        p, s = px[:k], sz[:k]
        empty = p < 0
        for i in range(levels):
            cols[f"{side}_px_{i + 1}"] = np.where(empty[:, i], np.nan, (p[:, i] + lo_tick) / 100.0)
            cols[f"{side}_sz_{i + 1}"] = np.where(empty[:, i], np.nan, s[:, i].astype(float))
    df = pd.concat([df, pd.DataFrame(cols)[book_columns(levels)]], axis=1)
    return _finish(classify_hidden_trades(df), levels)


def find_itch_days(raw_dir, symbol: str) -> list[tuple[str, Path]]:
    """[(date, npz_path), ...] written by scripts/extract_itch.py."""
    return sorted((p.parent.name, p) for p in Path(raw_dir).glob(f"*/{symbol}.npz"))
