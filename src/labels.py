"""Forward-return labels on the analysis grid.

r_{t,h} = log(mid_{t+h} / mid_t), where mid_{t+h} is the grid mid exactly
h ms ahead. Lookups are row-shifts validated against actual timestamps:
if the row k steps ahead is not exactly h ms ahead (grid gap from stale/
missing book data), the label is NaN rather than silently wrong.

Directional label: y_{t,h} = 1[r_{t,h} > 0] (NaN rows excluded downstream).

Also attaches execution reference prices at t+latency and t+latency+h used
by the backtest (best bid/ask prevailing at those future grid times).

Usage:
    python -m src.labels [--horizons-s 1 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def forward_lookup(ts: np.ndarray, arr: np.ndarray, ahead_ms: int, grid_ms: int) -> np.ndarray:
    """Value of `arr` exactly `ahead_ms` in the future; NaN when the grid
    row that far ahead does not exist at exactly that timestamp."""
    k = ahead_ms // grid_ms
    if k <= 0 or ahead_ms % grid_ms != 0:
        raise ValueError("ahead_ms must be a positive multiple of grid_ms")
    out = np.full(len(arr), np.nan)
    if k < len(arr):
        ok = (ts[k:] - ts[:-k]) == ahead_ms
        out[:-k] = np.where(ok, arr[k:], np.nan)
    return out


def add_labels(feats: pd.DataFrame, horizons_s: list[int], grid_ms: int) -> pd.DataFrame:
    feats = feats.copy()
    ts = feats["ts_ms"].to_numpy(dtype=np.int64)
    mid = feats["mid"].to_numpy()
    for h in horizons_s:
        h_ms = h * 1000
        fmid = forward_lookup(ts, mid, h_ms, grid_ms)
        feats[f"fwd_ret_{h}s"] = np.log(fmid / mid)
        feats[f"fwd_up_{h}s"] = np.where(np.isnan(fmid), np.nan, (fmid > mid).astype(float))
    return feats


def infer_grid_ms(feats: pd.DataFrame) -> int:
    return int(pd.Series(np.diff(feats["ts_ms"].to_numpy())).mode().iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    pdir = Path(args.processed_dir)
    feats = pd.read_parquet(pdir / "features.parquet")
    grid_ms = infer_grid_ms(feats)
    feats = add_labels(feats, args.horizons_s, grid_ms)
    feats.to_parquet(pdir / "dataset.parquet", index=False)
    n_ok = int(feats[f"fwd_ret_{args.horizons_s[0]}s"].notna().sum())
    print(f"dataset: {len(feats)} rows, {n_ok} with valid {args.horizons_s[0]}s label (grid={grid_ms}ms)")


if __name__ == "__main__":
    main()
