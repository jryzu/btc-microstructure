"""Tests for research-critical feature arithmetic."""
import numpy as np
import pandas as pd
import pytest

from src.features import book_state_features, make_grid, trade_window_features


def make_book_row(bid=100.0, ask=100.1, bid_qty=2.0, ask_qty=1.0, ts=1000):
    row = {"recv_ts_ms": ts, "last_update_id": ts}
    for i in range(20):
        row[f"bid_px_{i}"] = bid - 0.1 * i
        row[f"bid_qty_{i}"] = bid_qty
        row[f"ask_px_{i}"] = ask + 0.1 * i
        row[f"ask_qty_{i}"] = ask_qty
    return row


def test_mid_and_spread():
    book = pd.DataFrame([make_book_row(bid=100.0, ask=100.2)])
    f = book_state_features(book)
    assert f["mid"].iloc[0] == pytest.approx(100.1)
    assert f["spread"].iloc[0] == pytest.approx(0.2)
    assert f["spread_bps"].iloc[0] == pytest.approx(0.2 / 100.1 * 1e4)


def test_microprice_weights_toward_heavy_side():
    # bid size 3, ask size 1: buying pressure -> microprice above mid
    book = pd.DataFrame([make_book_row(bid=100.0, ask=100.1, bid_qty=3.0, ask_qty=1.0)])
    f = book_state_features(book)
    micro_expected = (100.0 * 1.0 + 100.1 * 3.0) / 4.0
    assert f["microprice"].iloc[0] == pytest.approx(micro_expected)
    assert f["microprice"].iloc[0] > f["mid"].iloc[0]


def test_microprice_equals_mid_when_balanced():
    book = pd.DataFrame([make_book_row(bid_qty=2.0, ask_qty=2.0)])
    f = book_state_features(book)
    assert f["microprice"].iloc[0] == pytest.approx(f["mid"].iloc[0])


def test_imbalance_levels():
    row = make_book_row(bid_qty=2.0, ask_qty=1.0)
    # make deeper levels balanced so imbalance_1 != imbalance_20
    for i in range(1, 20):
        row[f"bid_qty_{i}"] = 1.0
        row[f"ask_qty_{i}"] = 1.0
    f = book_state_features(pd.DataFrame([row]))
    assert f["imbalance_1"].iloc[0] == pytest.approx((2 - 1) / (2 + 1))
    q_b, q_a = 2.0 + 19.0, 1.0 + 19.0
    assert f["imbalance_20"].iloc[0] == pytest.approx((q_b - q_a) / (q_b + q_a))
    assert abs(f["imbalance_20"].iloc[0]) < abs(f["imbalance_1"].iloc[0])


def test_imbalance_bounds():
    book = pd.DataFrame([make_book_row(bid_qty=5.0, ask_qty=0.001)])
    f = book_state_features(book)
    for k in (1, 5, 10, 20):
        assert -1.0 <= f[f"imbalance_{k}"].iloc[0] <= 1.0


def test_grid_uses_only_past_snapshots():
    rows = [make_book_row(bid=100.0 + i, ask=100.1 + i, ts=1000 + 400 * i) for i in range(5)]
    book = pd.DataFrame(rows)
    bf = book_state_features(book)
    grid = make_grid(bf, grid_ms=500, max_staleness_ms=1000)
    assert (grid["book_ts_ms"] <= grid["ts_ms"]).all()


def test_grid_drops_stale_rows():
    rows = [make_book_row(ts=1000), make_book_row(ts=10000)]
    bf = book_state_features(pd.DataFrame(rows))
    grid = make_grid(bf, grid_ms=500, max_staleness_ms=1000)
    # times 2500..9500 have staleness > 1000ms relative to snapshot at 1000
    assert (grid["staleness_ms"] <= 1000).all()
    assert grid["ts_ms"].max() == 10000
    assert 5000 not in set(grid["ts_ms"])


def test_trade_window_excludes_future_trades():
    bf = book_state_features(pd.DataFrame([make_book_row(ts=1000), make_book_row(ts=3000)]))
    grid = make_grid(bf, grid_ms=500, max_staleness_ms=2500)
    trades = pd.DataFrame({
        "recv_ts_ms": [900, 1400, 2900],   # last trade is future for t=1000..2500
        "qty": [1.0, 2.0, 4.0],
        "is_buyer_maker": [False, True, False],
    })
    tf = trade_window_features(grid, trades)
    g = grid["ts_ms"].to_numpy()
    i = int(np.where(g == 1500)[0][0])
    # window (500, 1500]: buy of 1.0 at 900 and sell of 2.0 at 1400;
    # the future trade at 2900 must be excluded
    assert tf["vol_1s"].iloc[i] == pytest.approx(3.0)
    assert tf["signed_vol_1s"].iloc[i] == pytest.approx(-1.0)
    j = int(np.where(g == 3000)[0][0])
    # window (2000, 3000]: only the 2900 buy of 4.0
    assert tf["signed_vol_1s"].iloc[j] == pytest.approx(4.0)


def test_signed_volume_convention():
    """is_buyer_maker=True means seller-initiated => negative signed volume."""
    bf = book_state_features(pd.DataFrame([make_book_row(ts=1000)]))
    grid = make_grid(bf, grid_ms=500, max_staleness_ms=1000)
    trades = pd.DataFrame({
        "recv_ts_ms": [950], "qty": [3.0], "is_buyer_maker": [True]})
    tf = trade_window_features(grid, trades)
    assert tf["signed_vol_1s"].iloc[0] == pytest.approx(-3.0)
