"""Final v2 figures into reports/figures/.

Five figures tell the whole study:
  1. replication.png     — per-fold dev ICs for key signals + holdout hourly ICs
  2. imbalance_curves.png— imbalance decile vs forward return, both venues (dev)
  3. leadlag_basis.png   — cross-venue lead/lag correlations + basis behavior
  4. maker_economics.png — baseline vs skewed quoting, wf vs holdout, both fill models
  5. cost_reality.png    — taker edge vs fee levels (zero-fee margin scan)

Usage:  python -m src.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
PDIR = ROOT / "data" / "processed"
HOLDOUT_TS = 1787989320000

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_replication(sig_spot: dict, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    keys = ["imbalance_1@1s", "perp_imbalance_1@1s", "ofi_1s@1s",
            "ret_past_1s@1s", "basis_perp_bps@1s", "flow_x_imb@1s"]
    labels = ["imbalance", "perp book\n(cross)", "OFI", "momentum 1s",
              "basis", "flow×imb\n(failed)"]
    for i, (k, lab) in enumerate(zip(keys, labels)):
        folds = sig_spot["signals"].get(k, {}).get("folds", [])
        ics = [f["ic"] for f in folds]
        ax.scatter([i] * len(ics), ics, s=28, alpha=0.75,
                   color="tab:blue" if np.mean(ics or [0]) > 0.02 else "tab:red")
        if ics:
            ax.plot([i - 0.22, i + 0.22], [np.mean(ics)] * 2, color="k", lw=2)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("per-fold OOS Spearman IC")
    ax.set_title("Dev-region replication: spot 1s (6h folds)")

    ax = axes[1]
    hp = results["holdout_predictive"]
    for key, color, lab in (("spot@1s", "tab:blue", "spot 1s"),
                            ("spot@5s", "tab:cyan", "spot 5s"),
                            ("perp@1s", "tab:orange", "perp 1s"),
                            ("perp@5s", "tab:red", "perp 5s")):
        ics = hp[key]["hourly_ics"]
        ax.plot(range(len(ics)), ics, marker="o", ms=3.5, lw=1.1,
                color=color, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.set_xlabel("holdout hour (07:42–19:42 UTC, 2026-08-29)")
    ax.set_ylabel("hourly Spearman IC (ridge)")
    ax.set_title("Pre-registered holdout: IC positive 13/13 hours")
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(FIG / "replication.png")
    plt.close(fig)


def fig_imbalance_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, venue in zip(axes, ("spot", "perp")):
        d = pd.read_parquet(PDIR / f"dataset_{venue}.parquet",
                            columns=["ts_ms", "imbalance_1", "fwd_ret_1s", "fwd_ret_5s"])
        d = d[d["ts_ms"] < HOLDOUT_TS]
        for h, color in ((1, "tab:blue"), (5, "tab:orange")):
            df = d[["imbalance_1", f"fwd_ret_{h}s"]].dropna()
            df["bin"] = pd.qcut(df["imbalance_1"], 10, labels=False, duplicates="drop")
            g = df.groupby("bin")
            xm = g["imbalance_1"].mean()
            ym = g[f"fwd_ret_{h}s"].mean() * 1e4
            ysem = g[f"fwd_ret_{h}s"].sem() * 1e4
            ax.errorbar(xm, ym, yerr=1.96 * ysem, marker="o", ms=3.5, lw=1.2,
                        capsize=2, label=f"h={h}s", color=color)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"{venue}: forward return by imbalance decile (dev)")
        ax.set_xlabel("level-1 imbalance (decile mean)")
        ax.set_ylabel("mean forward return (bps)")
        ax.legend(fontsize=8)
    fig.savefig(FIG / "imbalance_curves.png")
    plt.close(fig)


def fig_leadlag_basis(ll: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    ks = list(range(1, 11))
    for key, color, lab in (("perp_leads_spot", "tab:orange", "perp ret → future spot ret"),
                            ("spot_leads_perp", "tab:blue", "spot ret → future perp ret")):
        cs = [ll[key].get(f"k{k}", {}).get("corr") for k in ks]
        ax.plot([k * 0.5 for k in ks], cs, marker="o", ms=4, lw=1.2,
                color=color, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("lead (seconds)")
    ax.set_ylabel("correlation of 500ms returns")
    ax.set_title("Cross-venue lead/lag (dev, non-overlapping)")
    ax.legend(fontsize=8)

    ax = axes[1]
    spot = pd.read_parquet(PDIR / "features_spot.parquet", columns=["ts_ms", "mid"])
    perp = pd.read_parquet(PDIR / "features_perp.parquet", columns=["ts_ms", "mid"])
    m = spot.merge(perp, on="ts_ms", suffixes=("_s", "_p"))
    m = m[m["ts_ms"] < HOLDOUT_TS]
    basis = (m["mid_p"] / m["mid_s"] - 1) * 1e4
    ax.hist(basis, bins=120, color="tab:purple", alpha=0.8)
    b = ll["basis"]
    ax.set_title(f"Perp−spot basis (dev): mean {b['mean_bps']:.1f}bps, "
                 f"σ {b['std_bps']:.2f}, half-life ≈ {b['half_life_s']:.1f}s")
    ax.set_xlabel("basis (bps)")
    ax.set_ylabel("count of 500ms observations")
    fig.savefig(FIG / "leadlag_basis.png")
    plt.close(fig)


def fig_maker_economics(mk: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    groups = [("wf_baseline", "wf_best_variant", "walk-forward"),
              ("holdout_baseline", "holdout_best_variant", "holdout")]
    ax = axes[0]
    width = 0.35
    for gi, (bkey, vkey, glab) in enumerate(groups):
        for off, key, color, lab in ((0, bkey, "tab:gray", "symmetric"),
                                     (width, vkey, "tab:orange", "signal-skewed")):
            v = mk.get(key, {})
            ax.bar(gi + off, v.get("pnl_per_fill_bps", 0), width * 0.95,
                   color=color, label=lab if gi == 0 else None)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks([0.17, 1.17], ["walk-forward", "holdout"])
    ax.set_ylabel("net P&L per fill (bps)")
    ax.set_title("Conservative (through) fills, 2bps maker fee")
    ax.legend(fontsize=8)

    ax = axes[1]
    cats = [("through", "holdout_baseline", "holdout_best_variant"),
            ("touch", "holdout_baseline_touch", "holdout_best_variant_touch")]
    x = np.arange(2)
    for off, idx, color, lab in ((0, 1, "tab:gray", "symmetric"),
                                 (width, 2, "tab:orange", "signal-skewed")):
        vals = [mk.get(c[idx], {}).get("avg_markout_1s_bps", np.nan) for c in cats]
        ax.bar(x + off, vals, width * 0.95, color=color, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x + width / 2, ["through (conservative)", "touch (optimistic)"])
    ax.set_ylabel("avg 1s markout per fill (bps)")
    ax.set_title("Holdout adverse selection by fill model")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "maker_economics.png")
    plt.close(fig)


def fig_cost_reality(bt: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    scan = bt["1s"]["wf_zero_fee_scan"]
    thr = [row["threshold"] * 1e4 for row in scan if row["n_trades"] > 0]
    edge = [row["avg_net_bps"] for row in scan if row["n_trades"] > 0]
    ntr = [row["n_trades"] for row in scan if row["n_trades"] > 0]
    ax.plot(thr, edge, marker="o", ms=5, lw=1.4, color="tab:blue",
            label="gross edge per trade (zero fees)")
    for t, e, n in zip(thr, edge, ntr):
        ax.annotate(f"{n} trades", (t, e), textcoords="offset points",
                    xytext=(6, -11), fontsize=7, color="tab:blue")
    for fee, color in ((1.8, "tab:green"), (5.0, "tab:orange"), (10.0, "tab:red")):
        ax.axhline(2 * fee, color=color, lw=1.1, ls="--")
        ax.annotate(f"round-trip cost @ {fee:g}bps/side", (thr[0], 2 * fee),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("signal threshold (bps of predicted move)")
    ax.set_ylabel("bps per trade (log scale)")
    ax.set_title("Perp taker reality: best gross edge ~1bps vs 3.6–20bps round-trip costs")
    ax.legend(fontsize=8, loc="center right")
    fig.savefig(FIG / "cost_reality.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    sig_spot = json.loads((ROOT / "reports" / "signals_spot.json").read_text())
    ll = json.loads((ROOT / "reports" / "leadlag.json").read_text())

    fig_replication(sig_spot, results)
    fig_imbalance_curves()
    fig_leadlag_basis(ll)
    fig_maker_economics(results["maker"]["perp"])
    fig_cost_reality(results["backtest"]["perp_s_ridge"])
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
