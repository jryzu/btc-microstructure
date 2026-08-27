"""Tests for v2 mechanisms: OFI, EV-rule backtest, maker fills/accounting."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BookLookup, simulate
from src.features import ofi_events
from src.maker import simulate_maker


# ---------- OFI ----------

def _book(ts, pb, qb, pa, qa):
    return pd.DataFrame({"recv_ts_ms": ts, "bid_px_0": pb, "bid_qty_0": qb,
                         "ask_px_0": pa, "ask_qty_0": qa})


def test_ofi_bid_size_increase_is_positive():
    b = _book([0, 100], [100.0, 100.0], [2.0, 5.0], [100.1, 100.1], [3.0, 3.0])
    e = ofi_events(b)["ofi_event"].to_numpy()
    # bid px equal (>= and <=): +qb_new - qb_old = +3; ask unchanged: -qa_new + qa_old = 0
    assert e[1] == pytest.approx(3.0)


def test_ofi_ask_size_increase_is_negative():
    b = _book([0, 100], [100.0, 100.0], [2.0, 2.0], [100.1, 100.1], [3.0, 7.0])
    e = ofi_events(b)["ofi_event"].to_numpy()
    assert e[1] == pytest.approx(-4.0)


def test_ofi_bid_price_up_counts_full_new_size():
    b = _book([0, 100], [100.0, 100.1], [2.0, 6.0], [100.2, 100.2], [3.0, 3.0])
    e = ofi_events(b)["ofi_event"].to_numpy()
    # bid up: +6 (new size), old not subtracted; ask unchanged: 0
    assert e[1] == pytest.approx(6.0)


def test_ofi_masked_across_gap():
    b = _book([0, 10_000], [100.0, 100.1], [2.0, 6.0], [100.2, 100.2], [3.0, 3.0])
    e = ofi_events(b, max_step_ms=2000)["ofi_event"].to_numpy()
    assert e[1] == 0.0


# ---------- EV-rule taker backtest ----------

def flat_lookup(n=200, bid=100.0, ask=100.02):
    ts = np.arange(0, n * 100, 100, dtype=np.int64)
    return BookLookup(pd.DataFrame({"recv_ts_ms": ts,
                                    "bid_px_0": np.full(n, bid),
                                    "ask_px_0": np.full(n, ask)}))


def test_ev_threshold_blocks_uneconomic_trades():
    """Signal below the EV threshold -> no trades at all (flat is optimal)."""
    lookup = flat_lookup()
    preds = pd.DataFrame({"ts_ms": np.array([1000, 2000], dtype=np.int64),
                          "p": [3e-5, -3e-5]})  # 0.3 bps signals
    rt_cost = 2 * 5.0 / 1e4  # 5 bps fee/side -> 10 bps round trip
    res = simulate(preds, lookup, "p", rt_cost, 5.0, 0, 1000)
    assert res["n_trades"] == 0
    assert res["net_total_bps"] == 0.0


def test_long_only_suppresses_shorts():
    lookup = flat_lookup()
    preds = pd.DataFrame({"ts_ms": np.array([1000, 3000], dtype=np.int64),
                          "p": [1.0, -1.0]})
    res = simulate(preds, lookup, "p", 0.5, 0.0, 0, 500, long_only=True)
    assert res["n_trades"] == 1
    assert res["n_short"] == 0


# ---------- maker simulation ----------

def maker_grid(n=20, bid=100.0, ask=100.02, sig=0.0, grid_ms=500):
    ts = np.arange(0, n * grid_ms, grid_ms, dtype=np.int64)
    return pd.DataFrame({"ts_ms": ts, "best_bid": np.full(n, bid),
                         "best_ask": np.full(n, ask),
                         "mid": np.full(n, (bid + ask) / 2),
                         "s_ridge": np.full(n, sig)})


def trades_at(rows):
    return pd.DataFrame({"recv_ts_ms": np.array([r[0] for r in rows], dtype=np.int64),
                         "price": [r[1] for r in rows],
                         "is_buyer_maker": [r[2] for r in rows]})


def test_through_fill_requires_price_strictly_through():
    grid = maker_grid()
    # trade AT the bid: touch-model fill, but NOT through-model fill
    tr = trades_at([(250, 100.0, True)])
    r_thru = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, 500, "through")
    r_touch = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, 500, "touch")
    assert r_thru["n_fills"] == 0
    assert r_touch["n_fills"] == 1


def test_through_fill_on_trade_below_bid():
    grid = maker_grid()
    tr = trades_at([(250, 99.99, True)])
    r = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, 500, "through")
    assert r["n_fills"] == 1
    assert r["n_buy_fills"] == 1


def test_maker_round_trip_pnl_captures_spread_minus_fees():
    """Buy filled at bid, sell filled at ask, flat book: P&L = spread - 2 fees."""
    grid = maker_grid(n=20, bid=100.0, ask=100.02)
    tr = trades_at([(250, 99.99, True), (750, 100.03, False)])
    fee_bps = 1.0
    r = simulate_maker(grid, tr, "s_ridge", None, None, fee_bps, 500, "through")
    assert r["n_fills"] == 2
    assert r["final_inventory"] == 0
    spread_bps = 0.02 / 100.01 * 1e4
    expected = spread_bps - 2 * fee_bps  # both legs pay maker fee
    assert r["final_pnl_bps"] == pytest.approx(expected, rel=1e-3)


def test_signal_pull_suppresses_adverse_side():
    """Strong positive signal (price going up) must pull the ASK."""
    grid = maker_grid(sig=5e-4)
    tr = trades_at([(250, 100.03, False), (750, 100.03, False)])  # buyers lifting
    r = simulate_maker(grid, tr, "s_ridge", 1e-4, None, 0.0, 500, "through")
    assert r.get("n_sell_fills", 0) == 0
    assert r["quoted_ask_frac"] < 0.1
    r0 = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, 500, "through")
    assert r0["n_sell_fills"] > 0


def test_inventory_cap_stops_one_side():
    grid = maker_grid(n=30)
    # relentless sellers hitting through the bid every window
    tr = trades_at([(i * 500 + 250, 99.99, True) for i in range(30)])
    r = simulate_maker(grid, tr, "s_ridge", None, 3, 0.0, 500, "through")
    assert r["max_abs_inventory"] <= 3
    r0 = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, 500, "through")
    assert r0["max_abs_inventory"] > 3


def test_markout_sign_is_adverse_when_price_moves_through_buy():
    """Buy fill then mid drops -> negative markout (adverse selection)."""
    n, grid_ms = 20, 500
    ts = np.arange(0, n * grid_ms, grid_ms, dtype=np.int64)
    mid = np.where(ts >= 1000, 99.0, 100.01)
    grid = pd.DataFrame({"ts_ms": ts, "best_bid": mid - 0.01, "best_ask": mid + 0.01,
                         "mid": mid, "s_ridge": np.zeros(n)})
    tr = trades_at([(250, 99.98, True)])
    r = simulate_maker(grid, tr, "s_ridge", None, None, 0.0, grid_ms, "through")
    assert r["n_fills"] == 1
    assert r["avg_markout_1s_bps"] < -50
