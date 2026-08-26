"""Chronological modeling of forward mid-price direction.

Split: first 60% train, next 20% validation, final 20% test (by time).
Models per horizon:
  - unconditional baseline (predict train base rate);
  - imbalance-sign heuristic (level-1 imbalance > 0 -> up);
  - logistic regression on standardized features;
  - HistGradientBoostingClassifier (small fixed config, no search).

Metrics: accuracy, ROC AUC, log loss, information coefficient (Pearson and
Spearman of predicted P(up) vs realized forward return), base rate.
Trading thresholds are chosen on validation only (see backtest.py).

Usage:
    python -m src.models
Writes model metrics into reports/results.json and per-row val/test
predictions to data/processed/predictions_<h>s.parquet for the backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

FEATURES = [
    "imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20",
    "micro_dev_bps", "spread_bps",
    "trade_imb_1s", "trade_imb_5s", "trade_imb_30s",
    "signed_vol_1s", "signed_vol_5s",
    "n_trades_1s", "n_trades_5s",
    "ret_past_1s", "ret_past_5s",
    "rv_30s", "spread_chg_5s", "depth_chg_5s",
]


def chrono_split(n: int, train: float = 0.6, val: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i1 = int(n * train)
    i2 = int(n * (train + val))
    idx = np.arange(n)
    return idx[:i1], idx[i1:i2], idx[i2:]


def _metrics(y: np.ndarray, p: np.ndarray, fwd_ret: np.ndarray, stride: int = 1) -> dict:
    """stride > 1 additionally reports IC on every stride-th observation so
    that overlapping labels (horizon > grid step) don't inflate apparent
    significance via serial correlation."""
    out = {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, p > 0.5)),
    }
    if 0 < np.mean(y) < 1 and len(np.unique(p)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        out["ic_pearson"] = float(pearsonr(p, fwd_ret)[0])
        out["ic_spearman"] = float(spearmanr(p, fwd_ret)[0])
        if stride > 1 and len(y) // stride >= 30:
            ps, rs = p[::stride], fwd_ret[::stride]
            ic, pv = pearsonr(ps, rs)
            out["ic_pearson_nonoverlap"] = float(ic)
            out["ic_nonoverlap_pvalue"] = float(pv)
            out["n_nonoverlap"] = int(len(ps))
    return out


def _reg_metrics(pred: np.ndarray, fwd_ret: np.ndarray, stride: int = 1) -> dict:
    """Metrics for a predicted-return (regression) signal."""
    out = {"n": int(len(pred)),
           "pred_mean_bps": float(np.mean(pred) * 1e4),
           "pred_std_bps": float(np.std(pred) * 1e4),
           "realized_std_bps": float(np.std(fwd_ret) * 1e4)}
    if len(np.unique(pred)) > 1:
        out["ic_pearson"] = float(pearsonr(pred, fwd_ret)[0])
        out["ic_spearman"] = float(spearmanr(pred, fwd_ret)[0])
        if stride > 1 and len(pred) // stride >= 30:
            ic, pv = pearsonr(pred[::stride], fwd_ret[::stride])
            out["ic_pearson_nonoverlap"] = float(ic)
            out["ic_nonoverlap_pvalue"] = float(pv)
            out["n_nonoverlap"] = int(len(pred[::stride]))
    return out


def run_horizon(df: pd.DataFrame, h: int, seed: int = 7) -> tuple[dict, pd.DataFrame]:
    ycol, rcol = f"fwd_up_{h}s", f"fwd_ret_{h}s"
    d = df.dropna(subset=FEATURES + [ycol, rcol]).reset_index(drop=True)
    X = d[FEATURES].to_numpy()
    y = d[ycol].to_numpy().astype(int)
    r = d[rcol].to_numpy()
    tr, va, te = chrono_split(len(d))
    grid_ms = int(pd.Series(np.diff(d["ts_ms"].to_numpy())).mode().iloc[0])
    stride = max(1, h * 1000 // grid_ms)

    results: dict = {"n_total": int(len(d)),
                     "split_ts_ms": {"train_end": int(d["ts_ms"].iloc[tr[-1]]),
                                     "val_end": int(d["ts_ms"].iloc[va[-1]]),
                                     "test_end": int(d["ts_ms"].iloc[te[-1]])}}
    preds = {"ts_ms": d["ts_ms"], "set": np.where(np.isin(np.arange(len(d)), va), "val",
                                                  np.where(np.isin(np.arange(len(d)), te), "test", "train")),
             "y": y, "fwd_ret": r,
             "best_bid": d["best_bid"], "best_ask": d["best_ask"], "mid": d["mid"]}

    # 1. unconditional
    base = np.full(len(d), y[tr].mean())
    results["unconditional"] = {s: _metrics(y[ix], base[ix], r[ix], stride) for s, ix in (("val", va), ("test", te))}

    # 2. imbalance heuristic: monotone map of imbalance_1 to a pseudo-probability
    imb = d["imbalance_1"].to_numpy()
    p_imb = (imb + 1.0) / 2.0
    results["imbalance_heuristic"] = {s: _metrics(y[ix], p_imb[ix], r[ix], stride) for s, ix in (("val", va), ("test", te))}
    preds["p_imbalance"] = p_imb

    # 3. logistic regression
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    logit.fit(X[tr], y[tr])
    p_lr = logit.predict_proba(X)[:, 1]
    results["logistic"] = {s: _metrics(y[ix], p_lr[ix], r[ix], stride) for s, ix in (("val", va), ("test", te))}
    results["logistic"]["coefficients"] = dict(zip(
        FEATURES, [float(c) for c in logit.named_steps["logisticregression"].coef_[0]]))
    preds["p_logistic"] = p_lr

    # 4. gradient boosting
    gbt = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=False, random_state=seed)
    gbt.fit(X[tr], y[tr])
    p_gbt = gbt.predict_proba(X)[:, 1]
    results["gbt"] = {s: _metrics(y[ix], p_gbt[ix], r[ix], stride) for s, ix in (("val", va), ("test", te))}
    preds["p_gbt"] = p_gbt

    # 5. ridge regression on forward returns: predicted-return signal in
    # return units, directly comparable with round-trip trading costs
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X[tr], r[tr])
    pred_r = ridge.predict(X)
    results["ridge_ret"] = {s: _reg_metrics(pred_r[ix], r[ix], stride) for s, ix in (("val", va), ("test", te))}

    # symmetric-around-zero trading signals for the backtest
    preds["s_ridge"] = pred_r
    preds["s_imbalance"] = imb
    preds["s_logistic"] = p_lr - 0.5
    preds["s_gbt"] = p_gbt - 0.5

    return results, pd.DataFrame(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    pdir = Path(args.processed_dir)
    df = pd.read_parquet(pdir / "dataset.parquet")

    reports_path = ROOT / "reports" / "results.json"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    all_results.setdefault("models", {})

    for h in args.horizons_s:
        res, preds = run_horizon(df, h)
        all_results["models"][f"{h}s"] = res
        preds.to_parquet(pdir / f"predictions_{h}s.parquet", index=False)
        for name in ("unconditional", "imbalance_heuristic", "logistic", "gbt"):
            m = res[name]["test"]
            print(f"h={h}s {name:22s} test: acc={m['accuracy']:.4f} "
                  f"auc={m.get('roc_auc', float('nan')):.4f} ic={m.get('ic_pearson', float('nan')):.4f}")
        m = res["ridge_ret"]["test"]
        print(f"h={h}s {'ridge_ret':22s} test: ic={m.get('ic_pearson', float('nan')):.4f} "
              f"ic_nonoverlap={m.get('ic_pearson_nonoverlap', float('nan')):.4f} "
              f"pred_std={m['pred_std_bps']:.2f}bps")

    reports_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {reports_path}")


if __name__ == "__main__":
    main()
