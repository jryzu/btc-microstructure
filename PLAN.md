# PLAN — v2

v1 (complete, on `main`): imbalance predicts 1–5s BTC spot moves (AUC 0.74, IC 0.18)
but +0.5bps gross/trade dies against 10bps taker fees. v2 pushes on the weaknesses:
more data, perp futures, maker-side economics, and a systematic alpha search.

## Collection (started 2026-08-27 19:42 UTC)

- Spot AND USDS-M perp BTCUSDT: `depth20@100ms` + `aggTrade`, 48 h target,
  15-min rotation, auto-reconnect. Files: `{spot|perp}_BTCUSDT_*.jsonl.gz`
  (v1 files without venue prefix = spot).
- Perp depth carries exchange timestamps (E/T) — spot does not; documented.
- Regime coverage goal: US + Asia + Europe hours across ≥2 weekdays
  (weekend only if collection extends; acknowledge if absent).
- caffeinate -s guard; machine must stay on AC power.

## Build order (against accumulating data)

1. **preprocess v2**: venue-aware parsing (futures `b`/`a` keys, E/T), outputs
   `book_{venue}.parquet` / `trades_{venue}.parquet`.
2. **features v2**: same 500 ms grid & v1 definitions (unchanged params —
   replication, not re-tuning). Drop `micro_dev_bps` from MODEL features
   (collinear with imbalance_1: micro−mid = spread/2·I₁); add interpretable
   candidates: event OFI, trade-burst z-scores, 10/30 s momentum/reversal,
   liquidity state, flow×book interaction, and cross-venue basis/lead-lag
   features (perp return → spot and vice versa).
3. **signals module**: univariate signal-quality evaluation SEPARATE from
   trading: per-fold OOS Spearman IC on non-overlapping samples, sign
   consistency across chronological folds and regimes (time-of-day, vol
   terciles). Output: replication table for every candidate.
4. **models v2**: walk-forward (expanding window over ~4 h folds) for
   development; FINAL ~25% holdout untouched until the very end, evaluated once.
5. **backtest v2**: perp fee structure (maker 2 / taker 5 bps VIP0; sensitivity
   incl. BNB 1.8/4.5 and lower tiers). EV-rule thresholding: trade only when
   |predicted move| > round-trip cost + margin — "no trade" is a legal optimum,
   selection never touches the holdout. Long/short reported separately.
   Spot shorting treated as NOT frictionless (perp is the short venue).
6. **maker sim** (new, perp fees): join-best quoting on the 500 ms grid,
   reservation price r = mid + α·ŝ − γ·inv acting through quote pulls/skews
   (1-tick spread ⇒ no room inside). Conservative trade-through fill model
   (strict price-through; touch-fill as optimistic bracket), queue position
   explicitly unknowable. Variants: symmetric / signal-skew / inventory-aware.
   Metrics: fills, spread captured, 1 s/5 s post-fill markout (adverse
   selection), inventory, P&L net of maker fee, max DD.
7. **lead/lag + basis** (the one extension): perp↔spot return lead/lag at
   0.1–5 s on the shared clock; basis distribution/dislocations; funding
   history via REST for context.
8. **Tests** for every new research-critical mechanism (OFI, EV threshold,
   fill model, markout, inventory accounting).
9. **Writeups**: update README, research note, STATUS (+ "Alpha Candidates
   Ranked"), dashboard artifact; commit+push throughout.

## Milestones

- M1 (~+2 h data): v2 pipeline smoke-tested end-to-end on both venues.
- M2 (~+6 h): first replication pass (v1 params, new period) + signal table.
- M3 (+24 h): interim full analysis incl. maker sim; regime splits.
- M4 (+48 h): final run, holdout evaluated ONCE, writeups, dashboard, push.

## Honesty rules (unchanged)

Chronological only; leakage tests must pass; holdout touched once; no
parameter changes to flatter results; zero-trade outcomes are valid; negative
results are results.
