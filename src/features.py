"""Build the analysis grid and microstructure features from processed tables.

The analysis clock is the LOCAL receive timestamp (the spot partial-depth
stream carries no exchange timestamp). All features at grid time t use only
data received at or before t:

  - book state: last depth20 snapshot with recv_ts_ms <= t (dropped if
    staler than --max-staleness-ms);
  - trade windows: aggregate trades with recv_ts_ms in (t - w, t].

Feature definitions:
  mid        = (best_bid + best_ask) / 2
  spread_bps = (best_ask - best_bid) / mid * 1e4
  microprice = (bid_px0 * ask_qty0 + ask_px0 * bid_qty0) / (bid_qty0 + ask_qty0)
               (size-weighted toward the heavier side's opposite quote)
  micro_dev_bps = (microprice - mid) / mid * 1e4
  imbalance_k = (sum_k bid_qty - sum_k ask_qty) / (sum_k bid_qty + sum_k ask_qty)
  signed volume: aggressive buy = +qty (aggTrade m == False), sell = -qty.

Usage:
    python -m src.features [--grid-ms 500] [--max-staleness-ms 1000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMB_DEPTHS = (1, 5, 10, 20)
TRADE_WINDOWS_S = (1, 5, 30)


def book_state_features(book: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot price/depth features (vectorized)."""
    out = pd.DataFrame({"recv_ts_ms": book["recv_ts_ms"].to_numpy()})
    bb = book["bid_px_0"].to_numpy()
    ba = book["ask_px_0"].to_numpy()
    bq = book["bid_qty_0"].to_numpy()
    aq = book["ask_qty_0"].to_numpy()
    mid = (bb + ba) / 2.0
    out["best_bid"] = bb
    out["best_ask"] = ba
    out["mid"] = mid
    out["spread"] = ba - bb
    out["spread_bps"] = (ba - bb) / mid * 1e4
    micro = (bb * aq + ba * bq) / (bq + aq)
    out["microprice"] = micro
    out["micro_dev_bps"] = (micro - mid) / mid * 1e4
    for k in IMB_DEPTHS:
        qb = book[[f"bid_qty_{i}" for i in range(k)]].to_numpy().sum(axis=1)
        qa = book[[f"ask_qty_{i}" for i in range(k)]].to_numpy().sum(axis=1)
        out[f"imbalance_{k}"] = (qb - qa) / (qb + qa)
    out["depth_total_20"] = (
        book[[f"bid_qty_{i}" for i in range(20)]].to_numpy().sum(axis=1)
        + book[[f"ask_qty_{i}" for i in range(20)]].to_numpy().sum(axis=1)
    )
    return out


def make_grid(book_feats: pd.DataFrame, grid_ms: int, max_staleness_ms: int) -> pd.DataFrame:
    """Snap the last-known book state onto a regular grid."""
    t0 = int(np.ceil(book_feats["recv_ts_ms"].iloc[0] / grid_ms) * grid_ms)
    t1 = int(book_feats["recv_ts_ms"].iloc[-1])
    grid_ts = np.arange(t0, t1 + 1, grid_ms, dtype=np.int64)
    grid = pd.DataFrame({"ts_ms": grid_ts})
    snapped = pd.merge_asof(
        grid, book_feats.rename(columns={"recv_ts_ms": "book_ts_ms"}),
        left_on="ts_ms", right_on="book_ts_ms", direction="backward",
    )
    snapped["staleness_ms"] = snapped["ts_ms"] - snapped["book_ts_ms"]
    snapped = snapped[snapped["staleness_ms"] <= max_staleness_ms].reset_index(drop=True)
    return snapped


def trade_window_features(grid: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Rolling trade-flow features over windows ending at each grid time.

    Windows are (t - w, t] on the local receive clock. Implemented with
    cumulative sums + searchsorted (trades sorted by recv_ts_ms).
    """
    trades = trades.sort_values("recv_ts_ms", kind="stable")
    ts = trades["recv_ts_ms"].to_numpy(dtype=np.int64)
    qty = trades["qty"].to_numpy()
    signed = np.where(trades["is_buyer_maker"].to_numpy(), -qty, qty)
    c_qty = np.concatenate([[0.0], np.cumsum(qty)])
    c_signed = np.concatenate([[0.0], np.cumsum(signed)])
    c_n = np.arange(len(ts) + 1, dtype=np.float64)

    g = grid["ts_ms"].to_numpy(dtype=np.int64)
    out = {}
    hi = np.searchsorted(ts, g, side="right")
    for w in TRADE_WINDOWS_S:
        lo = np.searchsorted(ts, g - w * 1000, side="right")
        vol = c_qty[hi] - c_qty[lo]
        sv = c_signed[hi] - c_signed[lo]
        n = c_n[hi] - c_n[lo]
        out[f"signed_vol_{w}s"] = sv
        out[f"vol_{w}s"] = vol
        out[f"n_trades_{w}s"] = n
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"trade_imb_{w}s"] = np.where(vol > 0, sv / vol, 0.0)
            out[f"avg_trade_size_{w}s"] = np.where(n > 0, vol / n, 0.0)
    return pd.DataFrame(out, index=grid.index)


def dynamics_features(grid: pd.DataFrame, grid_ms: int) -> pd.DataFrame:
    """Past returns / realized vol / state changes. Uses only past rows.

    Row-shift lookbacks are validated against actual timestamps so that
    grid gaps (dropped stale rows) never silently corrupt a lookback.
    """
    ts = grid["ts_ms"].to_numpy(dtype=np.int64)
    mid = grid["mid"].to_numpy()
    out = pd.DataFrame(index=grid.index)
    logmid = np.log(mid)

    def lagged(arr: np.ndarray, lag_ms: int) -> np.ndarray:
        k = lag_ms // grid_ms
        lag = np.full(len(arr), np.nan)
        if k < len(arr):
            ok = (ts[k:] - ts[:-k]) == lag_ms
            lag[k:] = np.where(ok, arr[:-k], np.nan)
        return lag

    for lb_ms, name in ((1000, "1s"), (5000, "5s")):
        out[f"ret_past_{name}"] = logmid - np.log(lagged(mid, lb_ms))
    r1 = pd.Series(logmid).diff()
    # a "return" spanning a grid gap is not a one-step return; mask it so it
    # cannot contaminate the realized-vol window after a capture gap
    valid_step = np.concatenate([[False], np.diff(ts) == grid_ms])
    r1[~valid_step] = np.nan
    out["rv_30s"] = r1.rolling(int(30000 / grid_ms), min_periods=10).std().to_numpy()
    out["spread_chg_5s"] = grid["spread_bps"].to_numpy() - lagged(grid["spread_bps"].to_numpy(), 5000)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["depth_chg_5s"] = np.log(grid["depth_total_20"].to_numpy() / lagged(grid["depth_total_20"].to_numpy(), 5000))
    return out


def ofi_events(book: pd.DataFrame, max_step_ms: int = 2000) -> pd.DataFrame:
    """Event-based order-flow imbalance (Cont/Kukanov/Stoikov style) from
    consecutive top-of-book snapshots.

    e_n = 1[Pb_n >= Pb_{n-1}]*Qb_n - 1[Pb_n <= Pb_{n-1}]*Qb_{n-1}
        - 1[Pa_n <= Pa_{n-1}]*Qa_n + 1[Pa_n >= Pa_{n-1}]*Qa_{n-1}

    Snapshot pairs separated by more than max_step_ms (capture gaps) are
    masked to zero contribution.
    """
    ts = book["recv_ts_ms"].to_numpy(dtype=np.int64)
    pb, qb = book["bid_px_0"].to_numpy(), book["bid_qty_0"].to_numpy()
    pa, qa = book["ask_px_0"].to_numpy(), book["ask_qty_0"].to_numpy()
    e = np.zeros(len(book))
    if len(book) > 1:
        up_b = pb[1:] >= pb[:-1]
        dn_b = pb[1:] <= pb[:-1]
        dn_a = pa[1:] <= pa[:-1]
        up_a = pa[1:] >= pa[:-1]
        contrib = (up_b * qb[1:] - dn_b * qb[:-1]
                   - dn_a * qa[1:] + up_a * qa[:-1])
        contrib[(ts[1:] - ts[:-1]) > max_step_ms] = 0.0
        e[1:] = contrib
    return pd.DataFrame({"recv_ts_ms": ts, "ofi_event": e})


def ofi_window_features(grid: pd.DataFrame, ofi: pd.DataFrame,
                        windows_s: tuple = (1, 5)) -> pd.DataFrame:
    """Sum snapshot-level OFI events over (t-w, t] windows on the grid."""
    ts = ofi["recv_ts_ms"].to_numpy(dtype=np.int64)
    c = np.concatenate([[0.0], np.cumsum(ofi["ofi_event"].to_numpy())])
    g = grid["ts_ms"].to_numpy(dtype=np.int64)
    hi = np.searchsorted(ts, g, side="right")
    out = {}
    for w in windows_s:
        lo = np.searchsorted(ts, g - w * 1000, side="right")
        out[f"ofi_{w}s"] = c[hi] - c[lo]
    return pd.DataFrame(out, index=grid.index)


def regime_features(grid: pd.DataFrame, tf: pd.DataFrame, grid_ms: int) -> pd.DataFrame:
    """Trailing-normalized activity/liquidity state + interactions.
    All rolling stats are past-only (window ends at the current row)."""
    out = pd.DataFrame(index=grid.index)
    win = int(300_000 / grid_ms)  # 5 min
    n1 = tf["n_trades_1s"]
    mu, sd = n1.rolling(win, min_periods=60).mean(), n1.rolling(win, min_periods=60).std()
    out["burst_1s"] = ((n1 - mu) / sd.replace(0.0, np.nan)).to_numpy()
    depth = np.log(grid["depth_total_20"])
    dmu, dsd = depth.rolling(win, min_periods=60).mean(), depth.rolling(win, min_periods=60).std()
    out["depth_z"] = ((depth - dmu) / dsd.replace(0.0, np.nan)).to_numpy()
    # past-only reference: expanding median, so no full-sample statistic leaks
    exp_med = grid["spread_bps"].expanding(min_periods=60).median()
    out["wide_spread"] = (grid["spread_bps"] > exp_med).astype(float).where(exp_med.notna()).to_numpy()
    out["flow_x_imb"] = (tf["trade_imb_5s"] * grid["imbalance_1"]).to_numpy()
    return out


def build_features(book: pd.DataFrame, trades: pd.DataFrame,
                   grid_ms: int = 500, max_staleness_ms: int = 1000) -> pd.DataFrame:
    bf = book_state_features(book)
    grid = make_grid(bf, grid_ms, max_staleness_ms)
    tf = trade_window_features(grid, trades)
    dyn = dynamics_features(grid, grid_ms)
    ofi = ofi_window_features(grid, ofi_events(book))
    reg = regime_features(grid, tf, grid_ms)
    feats = pd.concat([grid, tf, dyn, ofi, reg], axis=1)
    # longer momentum lookbacks (validated row shifts, same mechanism as dyn)
    ts = feats["ts_ms"].to_numpy(dtype=np.int64)
    logmid = np.log(feats["mid"].to_numpy())
    for lb_ms, name in ((10_000, "10s"), (30_000, "30s")):
        k = lb_ms // grid_ms
        lag = np.full(len(feats), np.nan)
        if k < len(feats):
            ok = (ts[k:] - ts[:-k]) == lb_ms
            lag[k:] = np.where(ok, logmid[:-k], np.nan)
        feats[f"ret_past_{name}"] = logmid - lag
    return feats


def add_cross_venue(feats: pd.DataFrame, other: pd.DataFrame, prefix: str,
                    grid_ms: int) -> pd.DataFrame:
    """Attach the other venue's state at the SAME grid timestamp: basis and
    the other venue's trailing 1s return (both strictly backward-looking).
    Grids share 500 ms epoch boundaries, so an exact ts join is causal."""
    ots = other["ts_ms"].to_numpy(dtype=np.int64)
    ologmid = np.log(other["mid"].to_numpy())
    k = 1000 // grid_ms
    oret = np.full(len(other), np.nan)
    if k < len(other):
        ok = (ots[k:] - ots[:-k]) == 1000
        oret[k:] = np.where(ok, ologmid[k:] - ologmid[:-k], np.nan)
    o = pd.DataFrame({
        "ts_ms": ots,
        f"{prefix}_mid": other["mid"].to_numpy(),
        f"{prefix}_ret_1s": oret,
        f"{prefix}_imbalance_1": other["imbalance_1"].to_numpy(),
    })
    merged = feats.merge(o, on="ts_ms", how="left")
    merged[f"basis_{prefix}_bps"] = (merged[f"{prefix}_mid"] - merged["mid"]) / merged["mid"] * 1e4
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-ms", type=int, default=500)
    ap.add_argument("--max-staleness-ms", type=int, default=1000)
    ap.add_argument("--venues", nargs="+", default=["spot", "perp"])
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    pdir = Path(args.processed_dir)

    built: dict[str, pd.DataFrame] = {}
    for venue in args.venues:
        bpath = pdir / f"book_{venue}.parquet"
        if not bpath.exists():
            print(f"[features] no book for {venue}; skipping")
            continue
        book = pd.read_parquet(bpath)
        trades = pd.read_parquet(pdir / f"trades_{venue}.parquet")
        built[venue] = build_features(book, trades, args.grid_ms, args.max_staleness_ms)

    if "spot" in built and "perp" in built:
        spot = add_cross_venue(built["spot"], built["perp"], "perp", args.grid_ms)
        perp = add_cross_venue(built["perp"], built["spot"], "spot", args.grid_ms)
        built = {"spot": spot, "perp": perp}

    for venue, feats in built.items():
        feats.to_parquet(pdir / f"features_{venue}.parquet", index=False)
        print(f"features_{venue}: {len(feats)} rows x {feats.shape[1]} cols")


if __name__ == "__main__":
    main()
