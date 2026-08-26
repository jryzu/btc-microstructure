"""Cost-aware aggressive execution test.

Rule: at grid time t a model emits a signal s symmetric around zero
(predicted forward return for the ridge model; P(up) - 0.5 for the
classifiers; raw level-1 imbalance for the heuristic). If s > thr, buy at
the ask prevailing at t + latency and exit at the bid prevailing at
t + latency + h. If s < -thr, mirror short. One position at a time
(signals during an open trade are ignored), unit notional per trade.

Candidate thresholds are quantiles of |s| computed on the VALIDATION set
(so the grid adapts to each signal's scale); the threshold maximizing
validation net P&L (with >= 20 trades) is then applied unchanged to the
TEST set.

Execution prices are looked up from the raw snapshot stream (last depth20
snapshot at or before the execution timestamp), so latency need not align
with the analysis grid. Taker fee is charged on both legs.

Net simple return of a long:  bid_exit * (1 - fee) / (ask_entry * (1 + fee)) - 1
Net simple return of a short: bid_entry * (1 - fee) / (ask_exit * (1 + fee)) - 1

Fee/latency sensitivity grids are reported on the test set for the chosen
threshold.

Usage:
    python -m src.backtest [--model s_ridge] [--horizons-s 1 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FEE_BPS = 10.0          # Binance spot taker, standard tier
DEFAULT_LATENCY_MS = 100.0
FEE_GRID_BPS = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
LATENCY_GRID_MS = [0.0, 100.0, 250.0, 500.0, 1000.0]
THRESHOLD_QUANTILES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


class BookLookup:
    """As-of lookup of best bid/ask from raw snapshots."""

    def __init__(self, book: pd.DataFrame):
        b = book.sort_values("recv_ts_ms", kind="stable")
        self.ts = b["recv_ts_ms"].to_numpy(dtype=np.int64)
        self.bid = b["bid_px_0"].to_numpy()
        self.ask = b["ask_px_0"].to_numpy()

    def quotes_at(self, t_ms: np.ndarray, max_staleness_ms: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.ts, t_ms, side="right") - 1
        valid = (idx >= 0) & (t_ms - self.ts[np.clip(idx, 0, None)] <= max_staleness_ms)
        idx = np.clip(idx, 0, len(self.ts) - 1)
        bid = np.where(valid, self.bid[idx], np.nan)
        ask = np.where(valid, self.ask[idx], np.nan)
        return bid, ask


def simulate(preds: pd.DataFrame, lookup: BookLookup, signal_col: str,
             threshold: float, fee_bps: float, latency_ms: float,
             horizon_ms: int) -> dict:
    """Sequential one-position-at-a-time simulation. Returns summary stats.

    The signal is symmetric around zero: > threshold goes long,
    < -threshold goes short.
    """
    ts = preds["ts_ms"].to_numpy(dtype=np.int64)
    s = preds[signal_col].to_numpy()
    fee = fee_bps / 1e4

    sig = np.zeros(len(s), dtype=np.int8)
    sig[s > threshold] = 1
    sig[s < -threshold] = -1

    entry_t = ts + int(latency_ms)
    exit_t = entry_t + horizon_ms
    ebid, eask = lookup.quotes_at(entry_t)
    xbid, xask = lookup.quotes_at(exit_t)

    rets = []
    sides = []
    trade_ts = []
    busy_until = -1
    for i in range(len(ts)):
        if sig[i] == 0 or ts[i] < busy_until:
            continue
        if np.isnan(ebid[i]) or np.isnan(xbid[i]):
            continue
        if sig[i] == 1:
            net = xbid[i] * (1 - fee) / (eask[i] * (1 + fee)) - 1
            gross = xbid[i] / eask[i] - 1
        else:
            net = ebid[i] * (1 - fee) / (xask[i] * (1 + fee)) - 1
            gross = ebid[i] / xask[i] - 1
        rets.append((gross, net))
        sides.append(int(sig[i]))
        trade_ts.append(int(ts[i]))
        busy_until = exit_t[i]

    n = len(rets)
    if n == 0:
        return {"n_trades": 0, "threshold": threshold, "fee_bps": fee_bps,
                "latency_ms": latency_ms}
    arr = np.array(rets)
    gross_bps = arr[:, 0] * 1e4
    net_bps = arr[:, 1] * 1e4
    cum = np.cumsum(net_bps)
    peak = np.maximum.accumulate(cum)
    span_h = (ts[-1] - ts[0]) / 3.6e6
    return {
        "threshold": threshold, "fee_bps": fee_bps, "latency_ms": latency_ms,
        "n_trades": n,
        "n_long": int(sum(1 for s in sides if s == 1)),
        "n_short": int(sum(1 for s in sides if s == -1)),
        "trades_per_hour": float(n / span_h) if span_h > 0 else float("nan"),
        "gross_total_bps": float(gross_bps.sum()),
        "net_total_bps": float(net_bps.sum()),
        "avg_gross_bps": float(gross_bps.mean()),
        "avg_net_bps": float(net_bps.mean()),
        "win_rate_net": float((net_bps > 0).mean()),
        "max_drawdown_bps": float((peak - cum).max()),
        "trade_ts_ms": trade_ts,
        "net_bps_series": [float(x) for x in net_bps],
    }


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("trade_ts_ms", "net_bps_series")}


def run(model: str, horizons_s: list[int], fee_bps: float, latency_ms: float,
        processed_dir: Path) -> dict:
    book = pd.read_parquet(processed_dir / "book.parquet",
                           columns=["recv_ts_ms", "bid_px_0", "ask_px_0"])
    lookup = BookLookup(book)
    out: dict = {"model": model, "fee_bps": fee_bps, "latency_ms": latency_ms}

    for h in horizons_s:
        preds = pd.read_parquet(processed_dir / f"predictions_{h}s.parquet")
        val = preds[preds["set"] == "val"].reset_index(drop=True)
        test = preds[preds["set"] == "test"].reset_index(drop=True)
        h_ms = h * 1000

        # candidate thresholds from |signal| quantiles on validation only
        abs_s = np.abs(val[model].to_numpy())
        thresholds = sorted(set(float(np.quantile(abs_s, q)) for q in THRESHOLD_QUANTILES))
        val_scan = [
            _strip(simulate(val, lookup, model, thr, fee_bps, latency_ms, h_ms))
            for thr in thresholds
        ]
        eligible = [s for s in val_scan if s["n_trades"] >= 20]
        chosen = max(eligible, key=lambda s: s["net_total_bps"]) if eligible \
            else max(val_scan, key=lambda s: s.get("net_total_bps", -np.inf))
        thr = chosen["threshold"]

        test_res = simulate(test, lookup, model, thr, fee_bps, latency_ms, h_ms)
        fee_sens = [_strip(simulate(test, lookup, model, thr, f, latency_ms, h_ms))
                    for f in FEE_GRID_BPS]
        lat_sens = [_strip(simulate(test, lookup, model, thr, fee_bps, lat, h_ms))
                    for lat in LATENCY_GRID_MS]
        thr_test_scan = [_strip(simulate(test, lookup, model, t, fee_bps, latency_ms, h_ms))
                         for t in thresholds]

        out[f"{h}s"] = {
            "chosen_threshold": thr,
            "threshold_grid": thresholds,
            "validation_scan": val_scan,
            "test": test_res,
            "fee_sensitivity_test": fee_sens,
            "latency_sensitivity_test": lat_sens,
            "threshold_scan_test_diagnostic": thr_test_scan,
        }
        t = test_res
        print(f"h={h}s thr={thr:.3g}: trades={t['n_trades']} "
              f"avg_gross={t.get('avg_gross_bps', float('nan')):.2f}bps "
              f"avg_net={t.get('avg_net_bps', float('nan')):.2f}bps "
              f"net_total={t.get('net_total_bps', float('nan')):.1f}bps")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="s_ridge",
                    choices=["s_ridge", "s_gbt", "s_logistic", "s_imbalance"])
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    ap.add_argument("--latency-ms", type=float, default=DEFAULT_LATENCY_MS)
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    res = run(args.model, args.horizons_s, args.fee_bps, args.latency_ms,
              Path(args.processed_dir))

    reports_path = ROOT / "reports" / "results.json"
    all_results = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    all_results.setdefault("backtest", {})[args.model] = res
    reports_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {reports_path}")


if __name__ == "__main__":
    main()
