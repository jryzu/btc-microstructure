"""Univariate signal-quality evaluation, separate from trading P&L.

For each candidate signal and horizon this module reports, on the
DEVELOPMENT region only (first 75% of rows; the final 25% holdout is never
read here):

  - per-fold Spearman IC on non-overlapping samples (folds = 6 h UTC blocks);
  - mean IC, cross-fold t-stat (Fama-MacBeth style), sign consistency;
  - IC within volatility terciles and depth terciles (conditioning);
  - IC by time-of-day session (Asia 00-08, EU 08-14, US 14-22, late 22-24 UTC).

A signal is "replicating" when its per-fold ICs share a sign in >= 75% of
folds and the cross-fold t-stat is materially > 2. This module makes no
trading claims: economic viability is assessed separately in backtest/maker.

Usage:
    python -m src.signals [--venue spot]
Writes reports/signals_{venue}.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    "imbalance_1", "imbalance_5", "imbalance_20",
    "trade_imb_1s", "trade_imb_5s", "signed_vol_1s",
    "ofi_1s", "ofi_5s",
    "ret_past_1s", "ret_past_5s", "ret_past_10s", "ret_past_30s",
    "flow_x_imb",
]
CROSS_CANDIDATES = {  # venue -> columns that exist only after cross-merge
    "spot": ["perp_ret_1s", "basis_perp_bps", "perp_imbalance_1"],
    "perp": ["spot_ret_1s", "basis_spot_bps", "spot_imbalance_1"],
}
CONDITIONERS = ["rv_30s", "depth_z", "burst_1s"]
HOLDOUT_FRAC = 0.25
FOLD_MS = 6 * 3600 * 1000
SESSIONS = [("asia", 0, 8), ("eu", 8, 14), ("us", 14, 22), ("late", 22, 24)]


def dev_region(df: pd.DataFrame) -> pd.DataFrame:
    n_dev = int(len(df) * (1 - HOLDOUT_FRAC))
    return df.iloc[:n_dev].copy()


def _ic(sig: np.ndarray, ret: np.ndarray, stride: int) -> tuple[float, int]:
    m = ~(np.isnan(sig) | np.isnan(ret))
    s, r = sig[m][::stride], ret[m][::stride]
    if len(s) < 30 or len(np.unique(s)) < 3:
        return np.nan, len(s)
    return float(spearmanr(s, r).statistic), len(s)


def fold_ics(df: pd.DataFrame, col: str, h: int, stride: int) -> list[dict]:
    out = []
    for fold, g in df.groupby(df["ts_ms"] // FOLD_MS):
        ic, n = _ic(g[col].to_numpy(), g[f"fwd_ret_{h}s"].to_numpy(), stride)
        if not np.isnan(ic) and n >= 100:
            out.append({"fold": int(fold), "ic": ic, "n": n})
    return out


def summarize(folds: list[dict]) -> dict:
    if not folds:
        return {"n_folds": 0}
    ics = np.array([f["ic"] for f in folds])
    mean = float(ics.mean())
    t = float(mean / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 1 else np.nan
    dom = float(max((ics > 0).mean(), (ics < 0).mean()))
    return {"n_folds": len(ics), "mean_ic": mean, "t_stat": t,
            "sign_consistency": dom, "min_ic": float(ics.min()),
            "max_ic": float(ics.max()), "folds": folds}


def conditioned_ics(df: pd.DataFrame, col: str, h: int, stride: int) -> dict:
    out = {}
    for cond in CONDITIONERS:
        if cond not in df.columns:
            continue
        try:
            terc = pd.qcut(df[cond], 3, labels=["low", "mid", "high"])
        except ValueError:
            continue
        cells = {}
        for name in ("low", "mid", "high"):
            sub = df[terc == name]
            ic, n = _ic(sub[col].to_numpy(), sub[f"fwd_ret_{h}s"].to_numpy(), stride)
            cells[name] = {"ic": None if np.isnan(ic) else round(ic, 4), "n": n}
        out[cond] = cells
    return out


def session_ics(df: pd.DataFrame, col: str, h: int, stride: int) -> dict:
    hours = (df["ts_ms"] // 3_600_000) % 24
    out = {}
    for name, lo, hi in SESSIONS:
        sub = df[(hours >= lo) & (hours < hi)]
        ic, n = _ic(sub[col].to_numpy(), sub[f"fwd_ret_{h}s"].to_numpy(), stride)
        out[name] = {"ic": None if np.isnan(ic) else round(ic, 4), "n": n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="spot", choices=["spot", "perp"])
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    df = pd.read_parquet(Path(args.processed_dir) / f"dataset_{args.venue}.parquet")
    dev = dev_region(df)
    grid_ms = int(pd.Series(np.diff(dev["ts_ms"].to_numpy())).mode().iloc[0])

    candidates = CANDIDATES + [c for c in CROSS_CANDIDATES[args.venue] if c in dev.columns]
    report: dict = {"venue": args.venue, "n_dev_rows": len(dev),
                    "holdout_frac": HOLDOUT_FRAC,
                    "dev_span": [int(dev["ts_ms"].iloc[0]), int(dev["ts_ms"].iloc[-1])],
                    "signals": {}}
    for h in args.horizons_s:
        stride = max(1, h * 1000 // grid_ms)
        for col in candidates:
            if col not in dev.columns:
                continue
            folds = fold_ics(dev, col, h, stride)
            s = summarize(folds)
            s["conditioned"] = conditioned_ics(dev, col, h, stride)
            s["sessions"] = session_ics(dev, col, h, stride)
            report["signals"][f"{col}@{h}s"] = s

    out = ROOT / "reports" / f"signals_{args.venue}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"== {args.venue} dev region: {len(dev)} rows, {report['signals'] and 'signals:' or ''}")
    rows = []
    for key, s in report["signals"].items():
        if s.get("n_folds", 0) >= 2:
            rows.append((key, s["mean_ic"], s["t_stat"], s["sign_consistency"], s["n_folds"]))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"{'signal@horizon':28s} {'meanIC':>8s} {'t':>6s} {'sign%':>6s} {'folds':>5s}")
    for key, ic, t, sc, nf in rows:
        print(f"{key:28s} {ic:8.4f} {t:6.2f} {sc:6.0%} {nf:5d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
