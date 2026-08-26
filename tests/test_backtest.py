"""Execution-cost arithmetic and simulation-behavior tests."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BookLookup, simulate


def make_book(ts, bid, ask):
    return pd.DataFrame({"recv_ts_ms": ts, "bid_px_0": bid, "ask_px_0": ask})


def flat_book(n=100, start=0, step=100, bid=100.0, ask=100.02):
    ts = np.arange(start, start + n * step, step, dtype=np.int64)
    return make_book(ts, np.full(n, bid), np.full(n, ask))


def preds_frame(ts, p):
    return pd.DataFrame({"ts_ms": np.array(ts, dtype=np.int64), "p": np.array(p, float)})


def test_flat_market_long_loses_spread_and_fees():
    """With static quotes, a round trip must lose exactly spread + 2 fees."""
    lookup = BookLookup(flat_book(n=200))
    preds = preds_frame([1000], [1.0])  # one strong long signal
    res = simulate(preds, lookup, "p", threshold=0.2, fee_bps=10.0,
                   latency_ms=100, horizon_ms=1000)
    assert res["n_trades"] == 1
    fee = 10.0 / 1e4
    expected_net = 100.0 * (1 - fee) / (100.02 * (1 + fee)) - 1
    assert res["avg_net_bps"] == pytest.approx(expected_net * 1e4, abs=1e-6)
    expected_gross = 100.0 / 100.02 - 1
    assert res["avg_gross_bps"] == pytest.approx(expected_gross * 1e4, abs=1e-6)
    assert res["avg_net_bps"] < res["avg_gross_bps"] < 0


def test_zero_fee_zero_spread_is_breakeven():
    lookup = BookLookup(flat_book(bid=100.0, ask=100.0))
    preds = preds_frame([1000], [1.0])
    res = simulate(preds, lookup, "p", 0.2, 0.0, 0, 1000)
    assert res["avg_net_bps"] == pytest.approx(0.0, abs=1e-9)


def test_long_profits_when_price_rises_enough():
    ts = np.arange(0, 5000, 100, dtype=np.int64)
    bid = np.where(ts >= 2000, 101.0, 100.0)
    ask = bid + 0.02
    lookup = BookLookup(make_book(ts, bid, ask))
    preds = preds_frame([900], [1.0])  # entry ~1000, exit ~2000 at higher bid
    res = simulate(preds, lookup, "p", 0.2, 1.0, 100, 1000)
    assert res["n_trades"] == 1
    assert res["avg_net_bps"] > 0


def test_short_side_arithmetic():
    """Short in a falling market: sell bid at entry, buy ask at exit."""
    ts = np.arange(0, 5000, 100, dtype=np.int64)
    bid = np.where(ts >= 2000, 99.0, 100.0)
    ask = bid + 0.02
    lookup = BookLookup(make_book(ts, bid, ask))
    preds = preds_frame([900], [-1.0])
    res = simulate(preds, lookup, "p", 0.2, 0.0, 100, 1000)
    assert res["n_trades"] == 1
    assert res["n_short"] == 1
    expected = 100.0 / 99.02 - 1
    assert res["avg_net_bps"] == pytest.approx(expected * 1e4, abs=1e-6)


def test_one_position_at_a_time():
    """Signals arriving while a trade is open must be ignored."""
    lookup = BookLookup(flat_book(n=300))
    ts = list(range(1000, 6000, 500))  # signals every 500ms
    preds = preds_frame(ts, [1.0] * len(ts))
    res = simulate(preds, lookup, "p", 0.2, 0.0, 0, horizon_ms=2000)
    # each trade occupies 2000ms => at most span/2000 + 1 trades
    assert res["n_trades"] <= 3


def test_higher_latency_never_uses_past_prices():
    """Entry uses quotes prevailing at t+latency (the price AFTER the jump
    if the jump happens inside the latency window)."""
    ts = np.arange(0, 4000, 100, dtype=np.int64)
    bid = np.where(ts >= 1500, 105.0, 100.0)
    ask = bid + 0.02
    lookup = BookLookup(make_book(ts, bid, ask))
    preds = preds_frame([1400], [1.0])
    res = simulate(preds, lookup, "p", 0.2, 0.0, 200, 1000)
    # entry at t=1600 must pay the post-jump ask 105.02, not 100.02
    expected = 105.0 / 105.02 - 1
    assert res["avg_net_bps"] == pytest.approx(expected * 1e4, abs=1e-6)


def test_threshold_gates_trades():
    lookup = BookLookup(flat_book(n=200))
    preds = preds_frame([1000, 2000, 3000], [0.02, 0.10, -0.02])
    res = simulate(preds, lookup, "p", threshold=0.05, fee_bps=0.0,
                   latency_ms=0, horizon_ms=500)
    assert res["n_trades"] == 1  # only s=0.10 passes |s| > 0.05
