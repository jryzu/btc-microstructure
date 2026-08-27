"""Cost-aware aggressive (taker) execution test — v2.

Signal: the ridge model's predicted forward return (or a classifier signal
recentred at zero). Decision rule is expected-value based:

    trade long  if  s > breakeven + margin
    trade short if  s < -(breakeven + margin)

where breakeven = round-trip cost in return units (2 * fee + observed spread
cost) and margin is a safety buffer chosen on the walk-forward region from a
small grid. "No trade anywhere" is a perfectly legal optimum: if no margin
produces positive expected net P&L on the wf region, the chosen policy is to
stay flat, and that is reported as the result.

Venues and default fees (per side, VIP0, no BNB discount):
  - spot: taker 10 bps.  Shorting spot is NOT frictionless (borrow needed);
    spot runs report the long-only variant as the economically honest one.
  - perp: taker 5 bps. Longs and shorts are symmetric and legitimate.

Region discipline: all selection happens on the `wf` (walk-forward
out-of-sample) predictions. The final holdout is simulated only when
--final is passed, once, with the wf-chosen policy frozen.

Usage:
    python -m src.backtest --venue perp [--final]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VENUE_FEES = {  # taker bps per side: default + sensitivity grid
    "spot": {"default": 10.0, "grid": [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]},
    "perp": {"default": 5.0, "grid": [0.0, 1.0, 1.8, 2.5, 4.5, 5.0]},
}
MARGIN_GRID_BPS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
LATENCY_GRID_MS = [0.0, 100.0, 250.0, 500.0, 1000.0]
QUANTILE_DIAG = [0.5, 0.8, 0.9, 0.95, 0.98]


class BookLookup:
    def __init__(self, book: pd.DataFrame):
        b = book.sort_values("recv_ts_ms", kind="stable")
        self.ts = b["recv_ts_ms"].to_numpy(dtype=np.int64)
        self.bid = b["bid_px_0"].to_numpy()
        self.ask = b["ask_px_0"].to_numpy()

    def quotes_at(self, t_ms: np.ndarray, max_staleness_ms: int = 2000):
        idx = np.searchsorted(self.ts, t_ms, side="right") - 1
        valid = (idx >= 0) & (t_ms - self.ts[np.clip(idx, 0, None)] <= max_staleness_ms)
        idx = np.clip(idx, 0, len(self.ts) - 1)
        return (np.where(valid, self.bid[idx], np.nan),
                np.where(valid, self.ask[idx], np.nan))


def simulate(preds: pd.DataFrame, lookup: BookLookup, signal_col: str,
             threshold: float, fee_bps: float, latency_ms: float,
             horizon_ms: int, long_only: bool = False) -> dict:
    """One-position-at-a-time taker simulation; threshold in return units."""
    ts = preds["ts_ms"].to_numpy(dtype=np.int64)
    s = preds[signal_col].to_numpy()
    fee = fee_bps / 1e4

    sig = np.zeros(len(s), dtype=np.int8)
    sig[s > threshold] = 1
    if not long_only:
        sig[s < -threshold] = -1

    entry_t = ts + int(latency_ms)
    exit_t = entry_t + horizon_ms
    ebid, eask = lookup.quotes_at(entry_t)
    xbid, xask = lookup.quotes_at(exit_t)

    rets, sides, trade_ts = [], [], []
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
    base = {"threshold": threshold, "fee_bps": fee_bps, "latency_ms": latency_ms,
            "long_only": long_only, "n_trades": n}
    if n == 0:
        base["net_total_bps"] = 0.0
        return base
    arr = np.array(rets)
    gross_bps, net_bps = arr[:, 0] * 1e4, arr[:, 1] * 1e4
    cum = np.cumsum(net_bps)
    peak = np.maximum.accumulate(cum)
    span_h = (ts[-1] - ts[0]) / 3.6e6
    long_mask = np.array(sides) == 1
    base.update({
        "n_long": int(long_mask.sum()), "n_short": int(n - long_mask.sum()),
        "trades_per_hour": float(n / span_h) if span_h > 0 else np.nan,
        "gross_total_bps": float(gross_bps.sum()),
        "net_total_bps": float(net_bps.sum()),
        "avg_gross_bps": float(gross_bps.mean()),
        "avg_net_bps": float(net_bps.mean()),
        "avg_net_long_bps": float(net_bps[long_mask].mean()) if long_mask.any() else None,
        "avg_net_short_bps": float(net_bps[~long_mask].mean()) if (~long_mask).any() else None,
        "win_rate_net": float((net_bps > 0).mean()),
        "max_drawdown_bps": float((peak - cum).max()),
        "trade_ts_ms": trade_ts,
        "net_bps_series": [float(x) for x in net_bps],
    })
    return base


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("trade_ts_ms", "net_bps_series")}


def run(venue: str, model: str, horizons_s: list[int], fee_bps: float,
        latency_ms: float, processed_dir: Path, final: bool) -> dict:
    book = pd.read_parquet(processed_dir / f"book_{venue}.parquet",
                           columns=["recv_ts_ms", "bid_px_0", "ask_px_0"])
    lookup = BookLookup(book)
    long_only = venue == "spot"
    out: dict = {"venue": venue, "model": model, "fee_bps": fee_bps,
                 "latency_ms": latency_ms, "long_only": long_only,
                 "policy": "EV-rule: |s| > 2*fee + margin (margin from wf)"}

    for h in horizons_s:
        preds = pd.read_parquet(processed_dir / f"predictions_{venue}_{h}s.parquet")
        wf = preds[preds["set"] == "wf"].reset_index(drop=True)
        h_ms = h * 1000
        rt_cost = 2 * fee_bps / 1e4  # spread cost enters via bid/ask prices

        # EV-margin selection on the walk-forward region only
        wf_scan = [_strip(simulate(wf, lookup, model, rt_cost + m / 1e4,
                                   fee_bps, latency_ms, h_ms, long_only))
                   for m in MARGIN_GRID_BPS]
        best = max(wf_scan, key=lambda s: s["net_total_bps"])
        trade_worthwhile = best["net_total_bps"] > 0 and best["n_trades"] >= 10
        chosen_thr = best["threshold"] if trade_worthwhile else None

        # diagnostics: quantile-threshold scan on wf (NOT used for selection)
        abs_s = np.abs(wf[model].to_numpy())
        diag = [_strip(simulate(wf, lookup, model, float(np.quantile(abs_s, q)),
                                fee_bps, latency_ms, h_ms, long_only))
                for q in QUANTILE_DIAG]

        block = {
            "wf_margin_scan": wf_scan,
            "wf_quantile_diagnostic": diag,
            "chosen_policy": ("no_trade" if not trade_worthwhile else
                              {"threshold": chosen_thr,
                               "margin_bps": (chosen_thr - rt_cost) * 1e4}),
        }

        # zero-fee wf diagnostic: is there ANY executable gross edge?
        zf = [_strip(simulate(wf, lookup, model, m / 1e4, 0.0, latency_ms,
                              h_ms, long_only)) for m in MARGIN_GRID_BPS]
        block["wf_zero_fee_scan"] = zf

        if final:
            hold = preds[preds["set"] == "holdout"].reset_index(drop=True)
            thr = chosen_thr if trade_worthwhile else None
            if thr is None:
                block["holdout"] = {"policy": "no_trade", "n_trades": 0,
                                    "net_total_bps": 0.0,
                                    "note": "wf region found no positive-EV policy; staying flat"}
            else:
                res = simulate(hold, lookup, model, thr, fee_bps, latency_ms, h_ms, long_only)
                block["holdout"] = res
                block["holdout_fee_sensitivity"] = [
                    _strip(simulate(hold, lookup, model, thr, f, latency_ms, h_ms, long_only))
                    for f in VENUE_FEES[venue]["grid"]]
                block["holdout_latency_sensitivity"] = [
                    _strip(simulate(hold, lookup, model, thr, fee_bps, lat, h_ms, long_only))
                    for lat in LATENCY_GRID_MS]
        out[f"{h}s"] = block

        pol = block["chosen_policy"]
        b = best
        print(f"{venue} h={h}s: wf best margin -> trades={b['n_trades']} "
              f"net_total={b['net_total_bps']:.1f}bps -> policy="
              f"{'NO TRADE' if pol == 'no_trade' else f'thr={chosen_thr:.3g}'}"
              + (f" | holdout: {block['holdout'].get('n_trades', 0)} trades, "
                 f"net={block['holdout'].get('net_total_bps', 0):.1f}bps" if final else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="perp", choices=["spot", "perp"])
    ap.add_argument("--model", default="s_ridge",
                    choices=["s_ridge", "s_gbt", "s_logistic", "s_imbalance"])
    ap.add_argument("--horizons-s", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--fee-bps", type=float, default=None)
    ap.add_argument("--latency-ms", type=float, default=100.0)
    ap.add_argument("--final", action="store_true",
                    help="evaluate the holdout ONCE with the wf-frozen policy")
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    fee = args.fee_bps if args.fee_bps is not None else VENUE_FEES[args.venue]["default"]

    res = run(args.venue, args.model, args.horizons_s, fee, args.latency_ms,
              Path(args.processed_dir), args.final)

    reports_path = ROOT / "reports" / "results.json"
    all_results = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    all_results.setdefault("backtest", {})[f"{args.venue}_{args.model}"] = res
    reports_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {reports_path}")


if __name__ == "__main__":
    main()
