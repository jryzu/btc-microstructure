"""Generate report figures into reports/figures/.

Descriptive analyses (binned imbalance/microprice plots, vol-regime split)
use the train+validation period only; test-set data appears only in model
evaluation and backtest figures.

Usage:
    python -m src.plots
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
FIG_DIR = ROOT / "reports" / "figures"
PDIR = ROOT / "data" / "processed"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})


def _binned(x: pd.Series, y: pd.Series, n_bins: int = 10):
    """Decile means of y by x with standard errors."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df["bin"] = pd.qcut(df["x"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin")
    return g["x"].mean(), g["y"].mean(), g["y"].sem(), g.size()


def fig_imbalance_vs_return(insample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, h in zip(axes, (1, 5)):
        for k, color in (("imbalance_1", "tab:blue"), ("imbalance_5", "tab:orange")):
            xm, ym, ysem, _ = _binned(insample[k], insample[f"fwd_ret_{h}s"] * 1e4)
            ax.errorbar(xm, ym, yerr=1.96 * ysem, marker="o", ms=3.5,
                        lw=1.2, capsize=2, label=k, color=color)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"Forward {h}s mid return by imbalance decile")
        ax.set_xlabel("order-book imbalance (decile mean)")
        ax.set_ylabel("mean forward return (bps)")
        ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "imbalance_vs_return.png")
    plt.close(fig)


def fig_microprice_and_vol_regimes(insample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    ax = axes[0]
    for h, color in ((1, "tab:blue"), (5, "tab:orange")):
        # milli-bps: BTCUSDT spread is ~1 tick, so deviations are tiny in bps
        xm, ym, ysem, _ = _binned(insample["micro_dev_bps"] * 1e3, insample[f"fwd_ret_{h}s"] * 1e4)
        ax.errorbar(xm, ym, yerr=1.96 * ysem, marker="o", ms=3.5, lw=1.2,
                    capsize=2, label=f"h={h}s", color=color)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Forward return by microprice deviation decile")
    ax.set_xlabel("microprice − mid (milli-bps, decile mean)")
    ax.set_ylabel("mean forward return (bps)")
    ax.legend(fontsize=8)

    ax = axes[1]
    med = insample["rv_30s"].median()
    for name, mask, color in (("low vol", insample["rv_30s"] <= med, "tab:green"),
                              ("high vol", insample["rv_30s"] > med, "tab:red")):
        sub = insample[mask]
        xm, ym, ysem, _ = _binned(sub["imbalance_1"], sub["fwd_ret_1s"] * 1e4)
        ax.errorbar(xm, ym, yerr=1.96 * ysem, marker="o", ms=3.5, lw=1.2,
                    capsize=2, label=name, color=color)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Imbalance vs 1s return by volatility regime")
    ax.set_xlabel("imbalance_1 (decile mean)")
    ax.set_ylabel("mean forward return (bps)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "microprice_and_vol_regimes.png")
    plt.close(fig)


def fig_calibration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, h in zip(axes, (1, 5)):
        preds = pd.read_parquet(PDIR / f"predictions_{h}s.parquet")
        test = preds[preds["set"] == "test"]
        for col, label, color in (("p_gbt", "gradient boosting", "tab:blue"),
                                  ("p_logistic", "logistic", "tab:orange")):
            bins = np.linspace(test[col].min(), test[col].max(), 11)
            idx = np.digitize(test[col], bins) - 1
            df = pd.DataFrame({"p": test[col], "y": test["y"], "b": idx})
            g = df.groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            g = g[g["n"] >= 30]
            ax.plot(g["p"], g["y"], marker="o", ms=4, lw=1.2, label=label, color=color)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="perfect")
        ax.set_title(f"Calibration, test set, h={h}s")
        ax.set_xlabel("predicted P(up)")
        ax.set_ylabel("realized frequency of up")
        ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "calibration.png")
    plt.close(fig)


def fig_cost_sensitivity(bt: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    ax = axes[0]
    for h, color in (("1s", "tab:blue"), ("5s", "tab:orange")):
        rows = bt[h]["fee_sensitivity_test"]
        fees = [r["fee_bps"] for r in rows]
        avg = [r.get("avg_net_bps", np.nan) for r in rows]
        ax.plot(fees, avg, marker="o", ms=4, lw=1.2, label=f"h={h}", color=color)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"Avg net edge per trade vs taker fee\n(latency={bt['latency_ms']:.0f}ms)")
    ax.set_xlabel("taker fee (bps per side)")
    ax.set_ylabel("avg net return per trade (bps)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for h, color in (("1s", "tab:blue"), ("5s", "tab:orange")):
        rows = bt[h]["latency_sensitivity_test"]
        lats = [r["latency_ms"] for r in rows]
        avg = [r.get("avg_net_bps", np.nan) for r in rows]
        ax.plot(lats, avg, marker="o", ms=4, lw=1.2, label=f"h={h}", color=color)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"Avg net edge per trade vs latency\n(fee={bt['fee_bps']:.0f}bps)")
    ax.set_xlabel("execution latency (ms)")
    ax.set_ylabel("avg net return per trade (bps)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "cost_sensitivity.png")
    plt.close(fig)


def fig_cumulative_pnl(bt: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    for h, color in (("1s", "tab:blue"), ("5s", "tab:orange")):
        t = bt[h]["test"]
        if t.get("n_trades", 0) == 0:
            continue
        ts = pd.to_datetime(np.array(t["trade_ts_ms"]), unit="ms")
        gross = None
        cum_net = np.cumsum(t["net_bps_series"])
        ax.plot(ts, cum_net, lw=1.2, color=color,
                label=f"h={h} net (thr={bt[h]['chosen_threshold']:.2g}, "
                      f"{t['n_trades']} trades)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"Cumulative net P&L on test set "
                 f"(fee={bt['fee_bps']:.0f}bps/side, latency={bt['latency_ms']:.0f}ms)")
    ax.set_ylabel("cumulative net P&L (bps of notional)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / "cumulative_pnl.png")
    plt.close(fig)


def fig_signal_decay_and_coefs(results: dict, insample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    ax = axes[0]
    horizons = sorted(results["models"].keys(), key=lambda s: int(s[:-1]))
    hs = [int(h[:-1]) for h in horizons]
    for model, color in (("gbt", "tab:blue"), ("logistic", "tab:orange"),
                         ("imbalance_heuristic", "tab:green")):
        ics = [results["models"][h][model]["test"].get("ic_pearson", np.nan) for h in horizons]
        ax.plot(hs, ics, marker="o", ms=5, lw=1.2, label=model, color=color)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(hs)
    ax.set_title("Signal decay: test IC by horizon")
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("IC (corr of P(up) with realized return)")
    ax.legend(fontsize=8)

    ax = axes[1]
    coefs = results["models"]["1s"]["logistic"]["coefficients"]
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(names, vals, color=["tab:red" if v < 0 else "tab:blue" for v in vals])
    ax.set_title("Logistic coefficients (standardized), h=1s")
    ax.set_xlabel("coefficient")
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(FIG_DIR / "signal_decay_and_coefficients.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    dataset = pd.read_parquet(PDIR / "dataset.parquet")
    val_end = results["models"]["1s"]["split_ts_ms"]["val_end"]
    insample = dataset[dataset["ts_ms"] <= val_end]

    fig_imbalance_vs_return(insample)
    fig_microprice_and_vol_regimes(insample)
    fig_calibration()
    bt_all = results.get("backtest", {})
    model = "s_ridge" if "s_ridge" in bt_all else next(iter(bt_all), None)
    if model:
        bt = results["backtest"][model]
        fig_cost_sensitivity(bt)
        fig_cumulative_pnl(bt)
    fig_signal_decay_and_coefs(results, insample)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
