"""Reusable falsification routines for candidate results.

These are deliberately generic: when a promising signal/strategy result
appears at a checkpoint, run it through this battery BEFORE believing it.

  - block_jackknife : influence of each time block on a statistic
                      (does one 30-min window drive the whole result?)
  - block_bootstrap_ci : CI for a mean via block bootstrap (respects
                      serial correlation better than iid resampling)
  - top_n_contribution : share of total P&L from the best N trades
                      (concentration = fragility)
  - split_stat      : statistic on both halves of an arbitrary boolean split
                      (session, vol regime, long/short, venue...)

All functions take plain numpy arrays; nothing here reads the holdout.
"""
from __future__ import annotations

import numpy as np


def block_jackknife(ts_ms: np.ndarray, values: np.ndarray, stat_fn,
                    block_ms: int = 1_800_000) -> dict:
    """Recompute stat_fn(values) with each time block removed.

    Returns the full-sample stat, per-block leave-one-out stats, and the
    max relative influence: |stat_full - stat_loo| / |stat_full|.
    """
    full = float(stat_fn(values))
    blocks = ts_ms // block_ms
    loo = {}
    for b in np.unique(blocks):
        keep = blocks != b
        if keep.sum() >= max(10, 0.2 * len(values)):
            loo[int(b)] = float(stat_fn(values[keep]))
    if not loo or full == 0:
        return {"stat_full": full, "n_blocks": len(loo), "max_influence": np.nan}
    infl = {b: abs(full - v) / abs(full) for b, v in loo.items()}
    worst = max(infl, key=infl.get)
    return {"stat_full": full, "n_blocks": len(loo),
            "max_influence": float(infl[worst]),
            "worst_block": int(worst),
            "stat_without_worst": loo[worst],
            "sign_flips_without_any_block": bool(any(np.sign(v) != np.sign(full)
                                                     for v in loo.values()))}


def block_bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                       block_len: int = 20, q: tuple = (0.025, 0.975),
                       seed: int = 11) -> dict:
    """CI for the mean of `values` using a moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < block_len * 2:
        return {"mean": float(np.mean(values)) if n else np.nan,
                "ci": [np.nan, np.nan], "note": "too few observations"}
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i][:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, q)
    return {"mean": float(np.mean(values)), "ci": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def top_n_contribution(pnl_series: np.ndarray, n: int = 5) -> dict:
    """What fraction of total P&L comes from the top-N winning trades?"""
    total = float(pnl_series.sum())
    top = np.sort(pnl_series)[-n:]
    return {"total": total, "top_n_sum": float(top.sum()),
            "top_n_share": float(top.sum() / total) if total != 0 else np.nan,
            "n_trades": int(len(pnl_series))}


def split_stat(values: np.ndarray, mask: np.ndarray, stat_fn) -> dict:
    """stat_fn on mask-true vs mask-false halves + agreement flag."""
    a = float(stat_fn(values[mask])) if mask.sum() >= 10 else np.nan
    b = float(stat_fn(values[~mask])) if (~mask).sum() >= 10 else np.nan
    return {"true_half": a, "false_half": b,
            "n_true": int(mask.sum()), "n_false": int((~mask).sum()),
            "signs_agree": bool(np.sign(a) == np.sign(b))
            if not (np.isnan(a) or np.isnan(b)) else None}
