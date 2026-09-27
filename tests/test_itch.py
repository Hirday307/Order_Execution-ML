"""ITCH 5.0 parsing and book rebuilding on a hand-built message stream."""
import gzip
import struct

import numpy as np
import pandas as pd
import pytest

from adaptive_exec.itch import extract_messages, itch_date, itch_to_events

SEC = 1_000_000_000
PRE, OPEN = 9 * 3600 * SEC, int(9.5 * 3600 * SEC)


def _hdr(t, loc, ts):
    return t.encode() + struct.pack(">HH", loc, 0) + ts.to_bytes(6, "big")


def _px(p):
    return int(round(p * 10_000))


def R(loc, sym):
    return (_hdr("R", loc, PRE - SEC) + sym.ljust(8).encode() + b"QN" + struct.pack(">I", 100)
            + b"NCZ PNN1N" + struct.pack(">I", 0) + b"N")


def S(ts):
    return _hdr("S", 0, ts) + b"O"


def A(loc, ts, ref, side, shares, sym, price):
    return _hdr("A", loc, ts) + struct.pack(">QcI8sI", ref, side.encode(), shares,
                                            sym.ljust(8).encode(), _px(price))


def E(loc, ts, ref, shares):
    return _hdr("E", loc, ts) + struct.pack(">QIQ", ref, shares, 1)


def C(loc, ts, ref, shares, printable, price):
    return _hdr("C", loc, ts) + struct.pack(">QIQcI", ref, shares, 2, printable.encode(), _px(price))


def X(loc, ts, ref, shares):
    return _hdr("X", loc, ts) + struct.pack(">QI", ref, shares)


def D(loc, ts, ref):
    return _hdr("D", loc, ts) + struct.pack(">Q", ref)


def U(loc, ts, ref, new_ref, shares, price):
    return _hdr("U", loc, ts) + struct.pack(">QQII", ref, new_ref, shares, _px(price))


def P(loc, ts, side, shares, sym, price):
    return _hdr("P", loc, ts) + struct.pack(">QcI8sIQ", 0, side.encode(), shares,
                                            sym.ljust(8).encode(), _px(price), 3)


def Q(loc, ts, shares, sym, price):
    return _hdr("Q", loc, ts) + struct.pack(">Q8sIQc", shares, sym.ljust(8).encode(), _px(price), 4, b"C")


def frame(msgs):
    return b"".join(struct.pack(">H", len(m)) + m for m in msgs)


LENGTHS = {"R": 39, "S": 12, "A": 36, "E": 31, "C": 36, "X": 23, "D": 19, "U": 35, "P": 44, "Q": 40}


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    t = lambda s: OPEN + int(s * SEC)
    msgs = [
        S(PRE - 2 * SEC), R(5, "MSFT"), R(9, "AAPL"),
        A(5, PRE, 1, "B", 300, "MSFT", 30.00),
        A(5, PRE, 2, "S", 200, "MSFT", 30.01),
        A(5, PRE, 3, "B", 100, "MSFT", 29.99),
        A(9, PRE, 4, "B", 999, "AAPL", 150.00),            # another stock: must be dropped
        E(5, t(1), 2, 50),                                  # buyer lifts 50 of the ask
        E(9, t(1), 4, 10),                                  # AAPL again
        X(5, t(2), 1, 100),                                 # 100 cancelled from the bid
        U(5, t(3), 3, 5, 100, 30.00),                       # 29.99 order moves up to 30.00
        P(5, t(4), "B", 70, "MSFT", 30.005),                # hidden trade at the midpoint: side unknown
        P(5, t(4.5), "B", 30, "MSFT", 30.01),               # hidden trade at the ask: buyer-initiated
        C(5, t(5), 2, 150, "N", 30.01),                     # executed in a cross, not printed
        A(5, t(6), 6, "S", 400, "MSFT", 30.02),
        D(5, t(7), 1),                                      # the rest of order 1 leaves
        Q(5, 16 * 3600 * SEC, 12345, "MSFT", 30.01),        # closing cross: after regular hours
    ]
    for m in msgs:
        assert len(m) == LENGTHS[chr(m[0])]
    d = tmp_path_factory.mktemp("itch")
    gz = d / "03272019.NASDAQ_ITCH50.gz"
    gz.write_bytes(gzip.compress(frame(msgs)))
    return d, gz


def test_extract_keeps_only_the_chosen_symbol_across_chunk_boundaries(stream):
    d, gz = stream
    counts = extract_messages(gz, ["MSFT"], d / "out", chunk_bytes=50, read_bytes=7, log=None)
    assert counts == {"MSFT": 12}
    m = np.load(d / "out" / "MSFT.npz")
    assert bytes(m["type"]).decode() == "AAAEXUPPCADQ"
    assert m["price"][0] == 300_000 and m["side"][0] == 1 and m["shares"][0] == 300
    assert m["ref2"][5] == 5 and m["shares"][11] == 12345
    assert itch_date(gz) == "2019-03-27"


def test_book_rebuild_emits_the_regular_hours_events(stream):
    d, gz = stream
    extract_messages(gz, ["MSFT"], d / "out2", log=None)
    ev = itch_to_events(d / "out2" / "MSFT.npz", "2019-03-27", levels=3)
    assert ev["ts"].iloc[0] == pd.Timestamp("2019-03-27 09:30:01")
    assert ev["type"].tolist() == [4, 2, 3, 1, 5, 5, 6, 1, 3]         # the replace is delete + add
    trade = ev.iloc[0]
    assert (trade["price"], trade["size"], trade["aggressor"], trade["is_trade"]) == (30.01, 50, 1, True)
    assert (trade["bid_px_1"], trade["bid_sz_1"], trade["bid_px_2"], trade["bid_sz_2"]) == (30.00, 300, 29.99, 100)
    assert (trade["ask_px_1"], trade["ask_sz_1"]) == (30.01, 150)
    assert ev["bid_sz_1"].iloc[1] == 200                             # after the partial cancel
    assert np.isnan(ev["bid_px_2"].iloc[2])                          # 29.99 emptied by the replace
    assert ev["bid_sz_1"].iloc[3] == 300                             # ...and re-added at 30.00
    at_mid, at_ask = ev.iloc[4], ev.iloc[5]                          # the 'P' side flag is ignored
    assert (at_mid["price"], at_mid["aggressor"], at_mid["is_trade"]) == (30.005, 0, True)
    assert (at_ask["price"], at_ask["aggressor"], at_ask["order_side"]) == (30.01, 1, -1)
    cross = ev.iloc[6]
    assert not cross["is_trade"] and np.isnan(cross["ask_px_1"]) and not cross["valid"]
    assert ev["ask_px_1"].iloc[7] == 30.02 and ev["valid"].iloc[7]
    assert (ev["size"].iloc[8], ev["bid_sz_1"].iloc[8]) == (200, 100)
    assert (ev["ts"] < pd.Timestamp("2019-03-27 16:00")).all()       # the closing cross is excluded
