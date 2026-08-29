# STATUS — v2 FINAL

_Completed 2026-08-29. Both collection phases done; pre-registered holdout
evaluated exactly once; all results pushed._

## Data collected

- **Spot BTCUSDT**: 1.67M top-20 snapshots (100ms) + 1.73M aggTrades; ≈46h clean.
- **Perp (USDS-M) BTCUSDT**: 1.48M snapshots + 5.01M trades; ≈44h clean.
- Periods: Aug-25 pilot (2.2h) + continuous Aug-27 19:42 → Aug-29 19:42 UTC;
  Asia/EU/US sessions; quiet and violent regimes (276k perp trades/15min burst).
- Known gaps NaN-masked: v1 machine-sleep gaps; 2.5h perp-feed outage Aug-28
  (fstream unreachable; collector self-recovered). Shared local clock
  (offset ~−25ms vs exchange, documented).
- 500ms analysis grids: 333,015 (spot) / 317,038 (perp) rows;
  holdout = 86.4k rows (12h), boundary declared 27h in advance.

## Experiments performed

1. Signal-quality study: 19 interpretable candidates × 2 venues × 2 horizons,
   per-6h-fold non-overlap Spearman IC, sign consistency, vol/depth/session
   conditioning (holdout excluded).
2. Walk-forward models (logistic/GBT direction; ridge return signal), final
   models evaluated once on the holdout.
3. Taker EV-rule backtest, venue-correct fees, no-trade admissible; frozen
   policy verified on holdout.
4. Maker quoting sim: symmetric vs signal-skew vs inventory-aware; through +
   touch fill brackets; 100ms quote latency; markouts at 0.5/1/5s;
   liquidation-adjusted P&L; frozen config on holdout.
5. Cross-venue lead/lag + basis study (non-overlapping returns).
6. Robustness: per-fold and per-hour replication, jackknife/bootstrap/
   concentration tooling, fee/latency sweeps.

## Strongest signals & replication

| Signal | Dev (folds) | Holdout | Replicated? |
|---|---|---|---|
| Book imbalance (own venue) | IC 0.31–0.39 (8/8 spot, 7/7 perp) | 0.31 univariate; ridge 13/13 hours | YES |
| Perp book → spot | IC 0.30–0.32 (7/7) | 0.26 | YES (asymmetric: reverse ≈0.24) |
| OFI | 0.23–0.28 (8/8) | 0.18 | YES |
| Momentum 1–30s | 0.10–0.26 (8/8) | via ridge | YES |
| Basis → spot convergence | 0.15–0.16 (7/7) | 0.17 | YES (tiny economics) |
| Flow×imbalance | ~0.00 (50–62% sign) | — | NO — failed |
| Basis → perp leg | −0.035 | — | NO — weak |

## Gross vs net economics (the answer)

- Taker: best gross ≈ +0.9–1.4bps/trade at strictest thresholds; round-trip
  costs 3.6bps (1.8bps/side BNB-VIP0 futures) → 20bps (spot standard).
  **EV-optimal policy = NO TRADE everywhere; held on holdout.**
- Maker (perp, 2bps fee, conservative fills): −3.1 to −3.4bps/fill regardless
  of skew; spread capture 0.007bps vs markout −0.5/−0.7/−1.0bps at 0.5/1/5s.
  Zero-fee through-model still −1.1bps/fill. **Passive quoting at the touch
  is adverse-selection-dominated at any accessible fee tier.**

## Maker results (holdout, frozen config thr=1.77e-5, cap=3)

- Through: symmetric −3.37bps/fill (4,808 fills) vs skewed −3.31 (2,207);
  1s markout −0.74 vs −0.87. Total loss halved via exposure reduction only.
- Touch (optimistic): markout improves −0.035→−0.025 but per-fill worsens
  −1.56→−2.03. **The 24h-checkpoint per-fill "quality improvement" did not
  replicate — downgraded.** Max |inventory| capped at 3 as designed.

## Major caveats

Two days, one exchange, one negative-basis regime, no weekend; holdout is
EU+US hours; fill models bracket unknowable queue position; absolute latency
approximate (clock offset); resting size would perturb the simulated queue.

## Alpha Candidates Ranked — FINAL

1. **Perp→spot information lead** — statistically robust everywhere;
   untradable directly at spot fees; genuine value as execution-timing
   overlay. Confidence: HIGH (information) / LOW (as standalone trade).
2. **Own-book imbalance/OFI prediction** — strongest, most replicated;
   priced in by fee structure; foundational input for any quoting engine.
   HIGH / LOW.
3. **Signal-gated maker exposure reduction** — halves adverse fills and total
   losses (both fill models, all periods); does NOT improve per-fill quality;
   cannot flip VIP0 sign. MEDIUM / LOW-MEDIUM (defensive only).
4. **Basis convergence via spot leg** — replicates; ~0.5bps moves vs ≥3.6bps
   costs. LOW.
5. **No tradable alpha at accessible fees** — the finding itself. HIGH.

## Resume bullets (final, numbers real)

1. Built a two-venue BTC microstructure research platform (live Binance spot +
   perpetual books/trades, 3.1M snapshots/6.7M trades over ~46h) with a
   pre-registered 12h holdout; demonstrated that order-book signals predict
   1–5s returns out-of-sample (Spearman IC 0.26–0.43, positive in 13/13
   holdout hours) and that the perpetual leads spot price discovery
   (IC 0.30 vs 0.24 reverse), while showing via EV-rule execution tests that
   no taker or VIP0 maker configuration monetizes the edge.
2. Designed leakage-resistant evaluation infrastructure for HFT signal
   research — walk-forward folds, frozen-policy holdout, conservative
   trade-through fill simulation with quote latency and markout-based
   adverse-selection measurement — which correctly falsified an in-sample
   maker improvement on out-of-sample data.

## 10 likely trader interview questions (answer notes)

1. *Why does imbalance predict?* Queue depletion + order splitting; effect is
   mechanical and universally known — which is exactly why it's priced.
2. *Why is your IC so high (0.3–0.4)? Suspicious?* It's Spearman on 500ms
   grids at 1s horizon — mostly predicting the next tick's direction, worth
   ~0.3bps. Huge IC, tiny magnitude; the two must be quoted together.
3. *Why believe the perp leads spot rather than clock artifacts?* One machine
   stamps both feeds; the asymmetry (0.157 vs 0.134 at 0.5s, persistence to
   2s vs 1.5s) survives non-overlapping sampling; and perp book state
   predicts spot even controlling for spot's own book.
4. *Your maker fill model?* Two bounds: strict trade-through (conservative,
   primary) and touch (optimistic, labeled). Queue position is unknowable
   publicly; I refuse to pretend otherwise — reality is between the bounds.
5. *Why did the maker skew fail?* Through-fills are definitionally the
   adversely-selected subset; pulling quotes removes benign and toxic fills
   almost equally, so per-fill quality doesn't improve — only exposure drops.
6. *What kills the taker trade — fees or latency?* Fees, by an order of
   magnitude (0.5bps edge vs ≥3.6bps round trip). Latency erodes the edge to
   zero in ~1s but never gets the chance to matter.
7. *One thing you'd distrust in your own study?* The single basis regime:
   perp sat 4bps under spot all week (negative carry). Lead/lag and basis
   results should be re-measured in a positive-carry week.
8. *Why 500ms bars, not event time?* Bursty 100ms snapshots (half arrive
   <20ms apart); 500ms halves autocorrelation while keeping 2 samples per
   1s horizon. Event-time bars are the obvious next refinement.
9. *What would make the maker experiment viable?* Rebate-tier fees, queue
   priors, and depth-aware quoting — quote where adverse selection is lower,
   not at the touch of a 1-tick book.
10. *What did the pre-registered holdout actually buy you?* It killed a
    result I believed (per-fill maker improvement) and confirmed the ones I
    doubted. That's the whole point.

## What works / what doesn't (engineering)

Works: dual-venue collector (session-proof nohup, auto-reconnect, health
monitor), full pipeline (preprocess→features→labels→signals→models→
backtest→maker→leadlag→plots), 38 tests, RUNBOOK-reproducible.
Not built: full diff-depth book reconstruction; event-time bars;
queue-position priors; rebate-tier maker scenarios (next).

## Exact next actions

1. Optional: extend capture across a weekend + positive-basis regime.
2. Implement execution-timing experiment (perp signal → spot parent order).
3. Rebate-tier maker sensitivity with queue priors.
