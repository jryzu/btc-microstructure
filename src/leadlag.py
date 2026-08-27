"""Cross-venue lead/lag and basis analysis (spot vs perp).

All statistics use NON-OVERLAPPING 500 ms returns on the shared local-clock
grid (both venues are timestamped by the same machine, so relative timing is
clean even though the absolute clock offset vs exchange time is unknown).

Reports, on the development region only (rows before the pre-registered
holdout boundary when given, else the first 75%):

  1. lead/lag cross-correlation: corr( ret_A[t], ret_B[t+k] ) for k = 1..10
     grid steps (0.5 s .. 5 s), both directions, with same-time corr k=0 as
     the contemporaneous baseline;
  2. basis distribution (perp/spot - 1, bps) and its mean-reversion speed
     (AR(1) coefficient on the 500 ms grid -> implied half-life);
  3. basis-extreme deciles -> forward convergence: after an unusually wide
     basis, does the gap close, and through which leg (spot moves vs perp)?

Usage:
    python -m src.leadlag [--holdout-start-ts 1787989320000]
Writes reports/leadlag.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def nonoverlap_returns(df: pd.DataFrame, grid_ms: int) -> pd.DataFrame:
    """500 ms log returns, NaN across grid gaps."""
    ts = df["ts_ms"].to_numpy(dtype=np.int64)
    logmid = np.log(df["mid"].to_numpy())
    ret = np.full(len(df), np.nan)
    ok = (ts[1:] - ts[:-1]) == grid_ms
    ret[1:] = np.where(ok, np.diff(logmid), np.nan)
    return pd.DataFrame({"ts_ms": ts, "ret": ret})


def leadlag_corr(a: pd.DataFrame, b: pd.DataFrame, grid_ms: int,
                 max_lead: int = 10) -> dict:
    """corr(ret_a[t], ret_b[t + k]) on the merged grid, k in [0, max_lead]."""
    m = a.merge(b, on="ts_ms", suffixes=("_a", "_b"))
    ra = m["ret_a"].to_numpy()
    rb = m["ret_b"].to_numpy()
    ts = m["ts_ms"].to_numpy(dtype=np.int64)
    out = {}
    for k in range(0, max_lead + 1):
        if k == 0:
            x, y, tvalid = ra, rb, np.ones(len(ra), bool)
        else:
            x, y = ra[:-k], rb[k:]
            tvalid = (ts[k:] - ts[:-k]) == k * grid_ms
        mask = tvalid & ~np.isnan(x) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 100:
            out[f"k{k}"] = {"corr": None, "n": n}
            continue
        c = float(np.corrcoef(x[mask], y[mask])[0, 1])
        # non-overlapping 500ms returns: se ~ 1/sqrt(n)
        out[f"k{k}"] = {"corr": round(c, 4), "n": n,
                        "t": round(c * np.sqrt(n), 1)}
    return out


def basis_analysis(spot: pd.DataFrame, perp: pd.DataFrame, grid_ms: int) -> dict:
    m = spot[["ts_ms", "mid"]].merge(perp[["ts_ms", "mid"]], on="ts_ms",
                                     suffixes=("_s", "_p"))
    ts = m["ts_ms"].to_numpy(dtype=np.int64)
    basis = (m["mid_p"] / m["mid_s"] - 1).to_numpy() * 1e4  # bps
    out = {"n": len(m),
           "mean_bps": float(np.mean(basis)), "std_bps": float(np.std(basis)),
           "p01_bps": float(np.quantile(basis, 0.01)),
           "p99_bps": float(np.quantile(basis, 0.99))}
    # AR(1) on consecutive grid points only
    ok = (ts[1:] - ts[:-1]) == grid_ms
    x, y = basis[:-1][ok], basis[1:][ok]
    if len(x) > 200 and np.std(x) > 0:
        phi = float(np.cov(x, y)[0, 1] / np.var(x))
        out["ar1_phi"] = phi
        out["half_life_s"] = float(-np.log(2) / np.log(abs(phi)) * grid_ms / 1000) \
            if 0 < abs(phi) < 1 else None
    # basis extremes -> which leg converges over the next 5 s
    k = 5000 // grid_ms
    valid = (ts[k:] - ts[:-k]) == 5000
    dbasis = np.full(len(basis), np.nan)
    ds = np.full(len(basis), np.nan)
    dp = np.full(len(basis), np.nan)
    dbasis[:-k] = np.where(valid, basis[k:] - basis[:-k], np.nan)
    logms = np.log(m["mid_s"].to_numpy())
    logmp = np.log(m["mid_p"].to_numpy())
    ds[:-k] = np.where(valid, (logms[k:] - logms[:-k]) * 1e4, np.nan)
    dp[:-k] = np.where(valid, (logmp[k:] - logmp[:-k]) * 1e4, np.nan)
    z = (basis - np.mean(basis)) / np.std(basis) if np.std(basis) > 0 else basis * 0
    cells = {}
    for name, mask in (("wide_high", z > 2), ("wide_low", z < -2),
                       ("normal", np.abs(z) <= 1)):
        mm = mask & ~np.isnan(dbasis)
        if mm.sum() >= 30:
            cells[name] = {"n": int(mm.sum()),
                           "d_basis_5s_bps": float(np.nanmean(dbasis[mm])),
                           "spot_move_5s_bps": float(np.nanmean(ds[mm])),
                           "perp_move_5s_bps": float(np.nanmean(dp[mm]))}
    out["extremes_5s"] = cells
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-start-ts", type=int, default=None)
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    pdir = Path(args.processed_dir)
    spot = pd.read_parquet(pdir / "features_spot.parquet", columns=["ts_ms", "mid"])
    perp = pd.read_parquet(pdir / "features_perp.parquet", columns=["ts_ms", "mid"])

    for df in (spot, perp):
        cut = args.holdout_start_ts or df["ts_ms"].iloc[int(len(df) * 0.75)]
        df.drop(df[df["ts_ms"] >= cut].index, inplace=True)

    grid_ms = int(pd.Series(np.diff(spot["ts_ms"].to_numpy())).mode().iloc[0])
    rs = nonoverlap_returns(spot, grid_ms)
    rp = nonoverlap_returns(perp, grid_ms)

    report = {
        "grid_ms": grid_ms,
        "note": "dev region only; shared local clock; non-overlapping returns",
        "perp_leads_spot": leadlag_corr(rp, rs, grid_ms),
        "spot_leads_perp": leadlag_corr(rs, rp, grid_ms),
        "basis": basis_analysis(spot, perp, grid_ms),
    }
    out = ROOT / "reports" / "leadlag.json"
    out.write_text(json.dumps(report, indent=2))
    for d, label in ((report["perp_leads_spot"], "perp->spot"),
                     (report["spot_leads_perp"], "spot->perp")):
        row = " ".join(f"k{k}={d[f'k{k}']['corr']}" for k in (0, 1, 2, 4)
                       if d.get(f"k{k}", {}).get("corr") is not None)
        print(f"{label}: {row}")
    b = report["basis"]
    print(f"basis: mean={b['mean_bps']:.2f}bps std={b['std_bps']:.2f} "
          f"half-life={b.get('half_life_s')}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
