"""The Databento loader on a real DBN file, written and decoded by Databento's own libraries."""
import pandas as pd
import pytest

db = pytest.importorskip("databento")
dbn = pytest.importorskip("databento_dbn")

from adaptive_exec.loaders import FILL_REMOVAL, load_databento  # noqa: E402

OPEN_UTC = pd.Timestamp("2026-03-02 14:30:00", tz="UTC")        # 09:30 New York (winter)


def _px(p):
    return int(round(p * 1e9))


def _book(bid, ask, bsz, asz):
    return [dbn.BidAskPair(bid_px=_px(bid - 0.01 * i), ask_px=_px(ask + 0.01 * i),
                           bid_sz=bsz, ask_sz=asz, bid_ct=1, ask_ct=1) for i in range(10)]


def test_mbp10_file_round_trip(tmp_path):
    t = OPEN_UTC + pd.Timedelta(seconds=5)
    s = pd.Timedelta(seconds=1)
    recs = [  # ts_event, recv offset (us), price, size, action, side, book after the event
        (OPEN_UTC - 60 * s, 1, 29.99, 10, "A", "B", _book(29.99, 30.01, 500, 400)),  # pre-open
        (t, 1, 30.00, 100, "A", "B", _book(30.00, 30.01, 600, 400)),
        (t + s, 1, 30.00, 200, "T", "A", _book(30.00, 30.01, 600, 400)),   # seller hits the bid
        (t + s, 2, 30.00, 200, "C", "B", _book(30.00, 30.01, 400, 400)),   # traded size leaves
        (t + 3 * s, 1, 30.01, 50, "C", "A", _book(30.00, 30.01, 400, 350)),  # a real cancel
    ]
    body = b"".join(bytes(dbn.MBP10Msg(
        publisher_id=2, instrument_id=1, ts_event=te.value, price=_px(p), size=sz,
        action=dbn.Action.from_str(a), side=dbn.Side.from_str(sd), depth=0,
        ts_recv=te.value + off * 1000, flags=128, ts_in_delta=0, sequence=i, levels=book))
        for i, (te, off, p, sz, a, sd, book) in enumerate(recs))
    meta = dbn.Metadata(dataset="XNAS.ITCH", start=(OPEN_UTC - 3600 * s).value,
                        stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
                        schema=dbn.Schema.MBP_10, symbols=["MSFT"], end=(OPEN_UTC + 7 * 3600 * s).value)
    path = tmp_path / "MSFT_2026-03-02.dbn"
    path.write_bytes(bytes(meta.encode()) + body)

    ev = load_databento(path)
    assert len(ev) == 4                                             # pre-open record dropped
    assert ev["ts"].iloc[0] == pd.Timestamp("2026-03-02 09:30:05.000001")
    assert ev["type"].tolist() == [1, 4, FILL_REMOVAL, 3]
    assert ev["aggressor"].tolist() == [0, -1, 0, 0]
    assert ev["price"].tolist() == [30.00, 30.00, 30.00, 30.01]
    assert ev["bid_sz_1"].tolist() == [600, 600, 400, 400]
    assert ev["bid_px_10"].iloc[0] == pytest.approx(29.91)
    assert ev["valid"].all()
