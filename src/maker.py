"""Passive market-making simulation with signal-skewed quoting — v2.

Question: even though the imbalance/flow signal is too small to pay taker
fees, does it have value for a MAKER — as a reason to pull or skew quotes to
avoid adverse selection?

Framework (per 500 ms decision window):
  reservation logic acts through DISCRETE quote decisions, because the
  BTCUSDT spread is one tick nearly always (no room to quote inside):
    - symmetric baseline: join best bid AND best ask every window;
    - signal skew:  s > +pull_thr  -> quote bid only (pull the ask that the
      signal says will be run over);  s < -pull_thr -> quote ask only;
    - inventory-aware: additionally stop quoting the side that would grow
      |inventory| beyond inv_cap.

Fill models (queue position is NOT reconstructable from public data — both
bounds are reported, reality is between them):
  - conservative "through": a resting buy at p fills only if a trade prints
    strictly BELOW p in the window (level traded through);
  - optimistic "touch": fills if a seller-initiated trade prints at <= p
    (assumes front of queue).

Accounting: unit size per fill, one fill max per side per window; maker fee
on notional both legs; equity marked to mid; markouts measured at +1 s/+5 s
vs mid (the adverse-selection metric).

Selection of (pull_thr, inv_cap) uses the wf region only; --final evaluates
the holdout once with the frozen configuration.

Usage:
    python -m src.maker --venue perp [--final]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MAKER_FEE_BPS = {"perp": 2.0, "spot": 10.0}   # VIP0, no BNB discount
FEE_SCENARIOS_BPS = [0.0, 1.0, 1.8, 2.0]
PULL_QUANTILES = [0.80, 0.90, 0.95]
INV_CAPS = [3, 10]


def _grid_forward_mid(ts: np.ndarray, mid: np.ndarray, ahead_ms: int, grid_ms: int) -> np.ndarray:
    k = ahead_ms // grid_ms
    out = np.full(len(mid), np.nan)
    if 0 < k < len(mid):
        ok = (ts[k:] - ts[:-k]) == ahead_ms
        out[:-k] = np.where(ok, mid[k:], np.nan)
    return out


def simulate_maker(grid: pd.DataFrame, trades: pd.DataFrame, signal_col: str,
                   pull_thr: float | None, inv_cap: int | None,
                   fee_bps: float, grid_ms: int, fill_model: str = "through") -> dict:
    """grid: ts_ms, best_bid, best_ask, mid, <signal_col>; trades sorted by recv."""
    ts = grid["ts_ms"].to_numpy(dtype=np.int64)
    bid_q = grid["best_bid"].to_numpy()
    ask_q = grid["best_ask"].to_numpy()
    mid = grid["mid"].to_numpy()
    s = grid[signal_col].to_numpy()
    fee = fee_bps / 1e4

    tts = trades["recv_ts_ms"].to_numpy(dtype=np.int64)
    tpx = trades["price"].to_numpy()
    tsell = trades["is_buyer_maker"].to_numpy()  # True = seller-initiated

    lo = np.searchsorted(tts, ts, side="left")
    hi = np.searchsorted(tts, ts + grid_ms, side="left")

    m1 = _grid_forward_mid(ts, mid, 1000, grid_ms)
    m5 = _grid_forward_mid(ts, mid, 5000, grid_ms)

    inv = 0
    cash = 0.0
    equity = np.full(len(ts), np.nan)
    fills = []  # (side, price, mid_now, m1, m5)
    windows_quoted_bid = windows_quoted_ask = 0
    pulls_bid = pulls_ask = 0

    for i in range(len(ts)):
        want_bid = want_ask = True
        if np.isnan(s[i]) or np.isnan(bid_q[i]):
            want_bid = want_ask = False
        else:
            if pull_thr is not None:
                if s[i] > pull_thr:
                    want_ask = False; pulls_ask += 1
                elif s[i] < -pull_thr:
                    want_bid = False; pulls_bid += 1
            if inv_cap is not None:
                if inv >= inv_cap:
                    want_bid = False
                if inv <= -inv_cap:
                    want_ask = False

        windows_quoted_bid += want_bid
        windows_quoted_ask += want_ask

        w = slice(lo[i], hi[i])
        if want_bid and hi[i] > lo[i]:
            if fill_model == "through":
                filled = bool((tpx[w] < bid_q[i]).any())
            else:  # touch
                filled = bool(((tpx[w] <= bid_q[i]) & tsell[w]).any())
            if filled:
                cash -= bid_q[i] * (1 + fee)
                inv += 1
                fills.append((1, bid_q[i], mid[i], m1[i], m5[i]))
        if want_ask and hi[i] > lo[i]:
            if fill_model == "through":
                filled = bool((tpx[w] > ask_q[i]).any())
            else:
                filled = bool(((tpx[w] >= ask_q[i]) & ~tsell[w]).any())
            if filled:
                cash += ask_q[i] * (1 - fee)
                inv -= 1
                fills.append((-1, ask_q[i], mid[i], m1[i], m5[i]))
        equity[i] = cash + inv * mid[i]

    n = len(fills)
    out = {"fill_model": fill_model, "fee_bps": fee_bps,
           "pull_thr": pull_thr, "inv_cap": inv_cap, "n_fills": n,
           "windows": int(len(ts)),
           "quoted_bid_frac": windows_quoted_bid / len(ts),
           "quoted_ask_frac": windows_quoted_ask / len(ts),
           "pulls_bid": pulls_bid, "pulls_ask": pulls_ask}
    if n == 0:
        out["final_pnl_bps"] = 0.0
        return out

    f = np.array([(side, px, m0, a, b) for side, px, m0, a, b in fills], dtype=float)
    side, px, m0, fm1, fm5 = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    # spread captured at fill: how far inside our fill was vs mid (bps)
    spread_cap = side * (m0 - px) / m0 * 1e4
    # markout: mid move against our inventory after the fill (bps); negative
    # markout = adverse selection (price moved through us)
    mk1 = side * (fm1 - m0) / m0 * 1e4
    mk5 = side * (fm5 - m0) / m0 * 1e4

    scale = np.nanmean(mid)
    eq_bps = (equity - equity[~np.isnan(equity)][0]) / scale * 1e4
    peak = np.fmax.accumulate(np.where(np.isnan(eq_bps), -np.inf, eq_bps))
    dd = np.nanmax(peak - eq_bps)

    out.update({
        "n_buy_fills": int((side == 1).sum()), "n_sell_fills": int((side == -1).sum()),
        "fills_per_hour": float(n / ((ts[-1] - ts[0]) / 3.6e6)),
        "avg_spread_captured_bps": float(np.nanmean(spread_cap)),
        "avg_markout_1s_bps": float(np.nanmean(mk1)),
        "avg_markout_5s_bps": float(np.nanmean(mk5)),
        "final_inventory": int(inv),
        "max_abs_inventory": int(np.max(np.abs(np.cumsum(side)))),
        "final_pnl_bps": float(eq_bps[~np.isnan(eq_bps)][-1]),
        "pnl_per_fill_bps": float(eq_bps[~np.isnan(eq_bps)][-1] / n),
        "max_drawdown_bps": float(dd),
        "equity_ts_ms": [int(t) for t in ts[:: max(1, len(ts) // 2000)]],
        "equity_bps": [None if np.isnan(x) else float(x)
                       for x in eq_bps[:: max(1, len(ts) // 2000)]],
    })
    return out


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("equity_ts_ms", "equity_bps")}


def run(venue: str, signal_h: int, latencyless: bool, processed_dir: Path,
        final: bool) -> dict:
    preds = pd.read_parquet(processed_dir / f"predictions_{venue}_{signal_h}s.parquet")
    trades = pd.read_parquet(processed_dir / f"trades_{venue}.parquet")
    trades = trades.sort_values("recv_ts_ms", kind="stable")
    fee = MAKER_FEE_BPS[venue]
    grid_ms = int(pd.Series(np.diff(preds["ts_ms"].to_numpy())).mode().iloc[0])

    wf = preds[preds["set"] == "wf"].reset_index(drop=True)
    out: dict = {"venue": venue, "signal": f"s_ridge@{signal_h}s",
                 "maker_fee_bps": fee, "grid_ms": grid_ms,
                 "note": "queue position unobservable; through/touch fills bracket reality"}

    abs_s = np.abs(wf["s_ridge"].to_numpy())
    thr_candidates = [None] + [float(np.nanquantile(abs_s, q)) for q in PULL_QUANTILES]

    # wf selection: symmetric baseline + skew/inventory variants
    scan = []
    for thr in thr_candidates:
        for cap in ([None] if thr is None else [None] + INV_CAPS):
            r = simulate_maker(wf, trades, "s_ridge", thr, cap, fee, grid_ms, "through")
            scan.append(_strip(r))
    out["wf_scan_through"] = scan
    baseline = next(r for r in scan if r["pull_thr"] is None)
    variants = [r for r in scan if r["pull_thr"] is not None]
    best = max(variants, key=lambda r: r["final_pnl_bps"]) if variants else None
    out["wf_baseline"] = baseline
    out["wf_best_variant"] = best

    # same configs under the optimistic touch model (bracket, not selection)
    out["wf_baseline_touch"] = _strip(simulate_maker(wf, trades, "s_ridge",
                                                     None, None, fee, grid_ms, "touch"))
    if best:
        out["wf_best_variant_touch"] = _strip(simulate_maker(
            wf, trades, "s_ridge", best["pull_thr"], best["inv_cap"], fee, grid_ms, "touch"))

    # fee sensitivity of baseline and best (wf, through)
    out["wf_fee_sensitivity"] = {
        "baseline": [_strip(simulate_maker(wf, trades, "s_ridge", None, None, f, grid_ms, "through"))
                     for f in FEE_SCENARIOS_BPS],
        "best": [_strip(simulate_maker(wf, trades, "s_ridge", best["pull_thr"],
                                       best["inv_cap"], f, grid_ms, "through"))
                 for f in FEE_SCENARIOS_BPS] if best else [],
    }

    if final and best:
        hold = preds[preds["set"] == "holdout"].reset_index(drop=True)
        out["holdout_baseline"] = simulate_maker(hold, trades, "s_ridge",
                                                 None, None, fee, grid_ms, "through")
        out["holdout_best_variant"] = simulate_maker(
            hold, trades, "s_ridge", best["pull_thr"], best["inv_cap"], fee, grid_ms, "through")
        out["holdout_baseline_touch"] = _strip(simulate_maker(
            hold, trades, "s_ridge", None, None, fee, grid_ms, "touch"))
        out["holdout_best_variant_touch"] = _strip(simulate_maker(
            hold, trades, "s_ridge", best["pull_thr"], best["inv_cap"], fee, grid_ms, "touch"))

    b, v = baseline, best
    print(f"{venue} maker wf (through, fee={fee}bps): baseline fills={b['n_fills']} "
          f"pnl={b['final_pnl_bps']:.1f}bps mk1s={b.get('avg_markout_1s_bps', float('nan')):.3f}")
    if v:
        print(f"  best variant thr={v['pull_thr']:.3g} cap={v['inv_cap']}: fills={v['n_fills']} "
              f"pnl={v['final_pnl_bps']:.1f}bps mk1s={v.get('avg_markout_1s_bps', float('nan')):.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="perp", choices=["perp", "spot"])
    ap.add_argument("--signal-horizon-s", type=int, default=1)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    res = run(args.venue, args.signal_horizon_s, True, Path(args.processed_dir), args.final)
    reports_path = ROOT / "reports" / "results.json"
    all_results = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    all_results.setdefault("maker", {})[args.venue] = res
    reports_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {reports_path}")


if __name__ == "__main__":
    main()
