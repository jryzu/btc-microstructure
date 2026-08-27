"""Walk-forward modeling of forward mid-price moves (v2).

Regions:
  - DEVELOPMENT: first 75% of rows, split into 5 sequential folds. For fold
    k = 2..5, models train on folds 1..k-1 and predict fold k, producing
    out-of-sample predictions over ~80% of the dev region ("wf" set). All
    threshold/parameter selection downstream uses ONLY these wf predictions.
  - HOLDOUT: final 25%. Models retrain once on the full dev region and
    predict it. The holdout is evaluated exactly once, at the end, by
    backtest/maker runs with parameters frozen from the wf region
    (--final flag there); it is never used for any selection.

Models per horizon: logistic + gradient boosting (direction), ridge
(returns — the economic signal), plus the raw imbalance heuristic.
micro_dev_bps is EXCLUDED from model features: it is (spread/2)·imbalance_1
by identity and was collinear with imbalance_1 in v1.

Usage:
    python -m src.models [--venue spot]
Writes wf metrics into reports/results.json and predictions (wf + holdout
rows tagged in `set`) to data/processed/predictions_{venue}_{h}s.parquet.
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_FRAC = 0.25
N_FOLDS = 5

BASE_FEATURES = [
    "imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20",
    "spread_bps",
    "trade_imb_1s", "trade_imb_5s", "trade_imb_30s",
    "signed_vol_1s", "signed_vol_5s",
    "n_trades_1s", "n_trades_5s",
    "ret_past_1s", "ret_past_5s", "ret_past_10s", "ret_past_30s",
    "rv_30s", "spread_chg_5s", "depth_chg_5s",
    "ofi_1s", "ofi_5s", "burst_1s", "depth_z", "flow_x_imb",
]
CROSS_FEATURES = {
    "spot": ["perp_ret_1s", "basis_perp_bps", "perp_imbalance_1"],
    "perp": ["spot_ret_1s", "basis_spot_bps", "spot_imbalance_1"],
}


def feature_list(venue: str, df: pd.DataFrame, use_cross: bool) -> list[str]:
    feats = [f for f in BASE_FEATURES if f in df.columns]
    if use_cross:
        feats += [f for f in CROSS_FEATURES[venue] if f in df.columns]
    return feats


def _cls_metrics(y, p, r, stride) -> dict:
    out = {"n": int(len(y)), "base_rate": float(np.mean(y)),
           "accuracy": float(accuracy_score(y, p > 0.5))}
    if 0 < np.mean(y) < 1 and len(np.unique(p)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        out["ic_pearson"] = float(pearsonr(p, r)[0])
        if stride > 1 and len(y) // stride >= 30:
            ic, pv = pearsonr(p[::stride], r[::stride])
            out["ic_nonoverlap"], out["ic_nonoverlap_p"] = float(ic), float(pv)
    return out


def _reg_metrics(pred, r, stride) -> dict:
    out = {"n": int(len(pred)), "pred_std_bps": float(np.std(pred) * 1e4)}
    if len(np.unique(pred)) > 1:
        out["ic_pearson"] = float(pearsonr(pred, r)[0])
        out["ic_spearman"] = float(spearmanr(pred, r).statistic)
        if stride > 1 and len(pred) // stride >= 30:
            ic, pv = pearsonr(pred[::stride], r[::stride])
            out["ic_nonoverlap"], out["ic_nonoverlap_p"] = float(ic), float(pv)
    return out


def _clip10(X):
    return np.clip(X, -10.0, 10.0)


def make_models(seed: int = 7) -> dict:
    # z-scores are clipped at +-10: features that are near-constant in a
    # short training fold (e.g. spread_chg when the spread is pinned at one
    # tick) otherwise produce colossal out-of-fold z-scores that blow up the
    # linear models.
    zclip = FunctionTransformer(_clip10)
    return {
        "logistic": make_pipeline(StandardScaler(), zclip,
                                  LogisticRegression(max_iter=2000, C=1.0)),
        "gbt": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              max_leaf_nodes=31, early_stopping=False,
                                              random_state=seed),
        "ridge": make_pipeline(StandardScaler(), zclip, Ridge(alpha=1.0)),
    }


def run_horizon(df: pd.DataFrame, h: int, venue: str, use_cross: bool) -> tuple[dict, pd.DataFrame]:
    ycol, rcol = f"fwd_up_{h}s", f"fwd_ret_{h}s"
    feats = feature_list(venue, df, use_cross)
    d = df.dropna(subset=feats + [ycol, rcol]).reset_index(drop=True)
    X = d[feats].to_numpy()
    y = d[ycol].to_numpy().astype(int)
    r = d[rcol].to_numpy()

    grid_ms = int(pd.Series(np.diff(d["ts_ms"].to_numpy())).mode().iloc[0])
    stride = max(1, h * 1000 // grid_ms)

    n_dev = int(len(d) * (1 - HOLDOUT_FRAC))
    bounds = np.linspace(0, n_dev, N_FOLDS + 1).astype(int)

    p_lr = np.full(len(d), np.nan)
    p_gbt = np.full(len(d), np.nan)
    pred_r = np.full(len(d), np.nan)
    setcol = np.array(["train"] * len(d), dtype=object)
    setcol[n_dev:] = "holdout"

    for k in range(1, N_FOLDS):
        tr = slice(0, bounds[k])
        te = slice(bounds[k], bounds[k + 1])
        m = make_models()
        m["logistic"].fit(X[tr], y[tr])
        m["gbt"].fit(X[tr], y[tr])
        m["ridge"].fit(X[tr], r[tr])
        p_lr[te] = m["logistic"].predict_proba(X[te])[:, 1]
        p_gbt[te] = m["gbt"].predict_proba(X[te])[:, 1]
        pred_r[te] = m["ridge"].predict(X[te])
        setcol[te] = "wf"

    # final models on the full dev region predict the untouched holdout
    m = make_models()
    m["logistic"].fit(X[:n_dev], y[:n_dev])
    m["gbt"].fit(X[:n_dev], y[:n_dev])
    m["ridge"].fit(X[:n_dev], r[:n_dev])
    hold = slice(n_dev, len(d))
    p_lr[hold] = m["logistic"].predict_proba(X[hold])[:, 1]
    p_gbt[hold] = m["gbt"].predict_proba(X[hold])[:, 1]
    pred_r[hold] = m["ridge"].predict(X[hold])

    wf = setcol == "wf"
    imb = d["imbalance_1"].to_numpy()
    results = {
        "n_total": int(len(d)), "n_dev": n_dev, "features": feats,
        "wf_span_ts_ms": [int(d['ts_ms'].to_numpy()[wf].min()), int(d['ts_ms'].to_numpy()[wf].max())],
        "imbalance_heuristic": {"wf": _cls_metrics(y[wf], (imb[wf] + 1) / 2, r[wf], stride)},
        "logistic": {"wf": _cls_metrics(y[wf], p_lr[wf], r[wf], stride)},
        "gbt": {"wf": _cls_metrics(y[wf], p_gbt[wf], r[wf], stride)},
        "ridge": {"wf": _reg_metrics(pred_r[wf], r[wf], stride)},
    }
    ridge_coefs = m["ridge"].named_steps["ridge"].coef_
    results["ridge"]["coefficients_final"] = dict(zip(feats, [float(c) for c in ridge_coefs]))

    preds = pd.DataFrame({
        "ts_ms": d["ts_ms"], "set": setcol, "y": y, "fwd_ret": r,
        "best_bid": d["best_bid"], "best_ask": d["best_ask"], "mid": d["mid"],
        "s_ridge": pred_r, "s_gbt": p_gbt - 0.5, "s_logistic": p_lr - 0.5,
        "s_imbalance": imb,
    })
    return results, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="spot", choices=["spot", "perp"])
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--no-cross", action="store_true",
                    help="exclude cross-venue features")
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    pdir = Path(args.processed_dir)
    df = pd.read_parquet(pdir / f"dataset_{args.venue}.parquet")

    reports_path = ROOT / "reports" / "results.json"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    all_results.setdefault("models", {}).setdefault(args.venue, {})

    for h in args.horizons_s:
        res, preds = run_horizon(df, h, args.venue, not args.no_cross)
        all_results["models"][args.venue][f"{h}s"] = res
        preds.to_parquet(pdir / f"predictions_{args.venue}_{h}s.parquet", index=False)
        for name in ("imbalance_heuristic", "logistic", "gbt"):
            m = res[name]["wf"]
            print(f"{args.venue} h={h}s {name:20s} wf: auc={m.get('roc_auc', float('nan')):.4f} "
                  f"ic={m.get('ic_pearson', float('nan')):.4f}")
        m = res["ridge"]["wf"]
        print(f"{args.venue} h={h}s {'ridge':20s} wf: ic={m.get('ic_pearson', float('nan')):.4f} "
              f"nonovl={m.get('ic_nonoverlap', float('nan')):.4f} pred_std={m['pred_std_bps']:.2f}bps")

    reports_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {reports_path}")


if __name__ == "__main__":
    main()
