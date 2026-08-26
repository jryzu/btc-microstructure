# STATUS

_Final. Last updated: 2026-08-26._

## What works

- **Collector** (`src/collect.py`): live Binance Spot combined websocket
  (`depth20@100ms` + `aggTrade`), gzipped JSONL with local receive
  timestamps, 15-min rotation, auto-reconnect. Ran 4 h against the real feed.
- **Pipeline**: preprocess (sanity filters + quality report) → 500 ms feature
  grid → timestamp-validated 1 s/5 s labels → chronological 60/20/20 models →
  cost-aware taker backtest with validation-only threshold selection →
  6 figures. Reruns end-to-end with the six commands in `RUNBOOK.md`.
- **Tests**: 21 passing (`pytest -q`) — imbalance/microprice/mid arithmetic,
  grid causality, trade-window future-exclusion, label alignment and an
  explicit leakage-injection test, execution-cost arithmetic, one-position
  constraint, latency lookup.

## What does not / was not built

- No market-making (passive fill) simulation — listed as the next experiment;
  the aggressive test answered the core question first.
- No multi-day or multi-venue data; no full diff-depth book reconstruction.
- 10 s horizon skipped (optional in the brief; 1 s vs 5 s already shows decay).

## Data collected

- 2026-08-25 18:46–22:55 UTC, Binance Spot BTCUSDT public streams.
- 80,587 top-20 snapshots + 109,563 aggregate trades; ~134 clean minutes
  (several machine-sleep gaps, incl. one 44 min; pipeline NaN-safe across gaps).
- BTC range $78,178–$79,237; spread = 1 tick in 99.9% of snapshots.
- Median trade receive latency 64 ms (p99 ~0.6 s); bursty snapshot arrivals.

## Key results (held-out chronological test)

- Imbalance decile → forward return: monotone, −0.34…+0.30 bps (1 s),
  −0.83…+0.65 bps (5 s).
- Direction AUC: 0.74 (1 s) from raw imbalance alone; models add ~nothing.
- Ridge return signal IC: 0.19 at 1 s (0.18 non-overlapping, p≈6e-13);
  0.12 (non-overlap) at 5 s.
- Backtest (98th-pct threshold, 100 ms latency): +0.50 bps avg gross/trade at
  1 s; −19.5 bps net at 10 bps/side taker fee; breakeven ≈ 0.25 bps/side;
  1,000 ms latency erases the gross edge entirely.
- Verdict: real signal, unexploitable via taker execution; value is in
  maker quote-skewing / adverse-selection avoidance.

## Major assumptions

- Local receive clock as analysis clock (spot depth has no exchange ts).
- Aggressor side from aggTrade `m` flag (buyer-is-maker ⇒ sell-initiated).
- Fills at displayed top-of-book quotes, no market impact, unit notional.
- Fee model: flat per-side taker bps; default 10 bps (standard tier).

## Known limitations

Single afternoon/venue/regime; short-dominated test trades (50/52) due to a
mild downtrend; overlapping labels (mitigated by non-overlap ICs);
partial-depth stream; capture gaps; no queue modeling.

## Exact next actions

1. Collect a multi-day sample (`python -m src.collect --duration-minutes 480`
   overnight, machine set to never sleep).
2. Implement reservation-price maker simulation with trade-through fills.
3. Add event-based OFI feature; compare with snapshot imbalance.
4. Push to GitHub (repo is committed and ready).

---

## Resume bullet options

1. Built an end-to-end BTC/USDT microstructure research pipeline on live
   Binance order-book/trade streams; showed order-book imbalance predicts
   1–5 s mid-price moves out-of-sample (AUC 0.74, non-overlapping IC 0.18,
   p≈1e-12) but that the ~0.5 bps gross edge is ~40× too small to survive
   taker fees — locating the signal's value in maker quote management.
2. Designed leakage-safe chronological evaluation of high-frequency crypto
   signals (timestamp-validated labels, leakage-injection tests,
   validation-only thresholding) and a cost-aware execution simulator with
   fee/latency sensitivity analysis on live-captured Binance data.
3. Built a websocket market-data capture and research stack (Python,
   pandas/sklearn, Parquet) for BTC/USDT top-of-book data; quantified signal
   decay (~1 s half-life) and breakeven fee (~0.25 bps/side) for an
   imbalance-based taker strategy, reporting an honest negative economic result.

## Interview talking points

1. **Why should imbalance predict price?** Queue depletion mechanics: a
   heavier bid queue means the ask side is consumed first; plus informed
   traders split orders, leaving footprints. It's the standard microprice
   logic — my data shows the effect is real but ~0.5 bps.
2. **What leakage risks existed and how did you handle them?** Future data in
   features (grid uses only snapshots ≤ t; trade windows are (t−w, t]);
   label misalignment across grid gaps (labels NaN unless the row exactly
   h ahead exists); threshold tuning on test (thresholds chosen on
   validation only). A test injects a future jump and asserts only labels move.
3. **Why a 500 ms grid?** Snapshots arrive ~100 ms but bursty; 500 ms gives
   ≥2 samples per 1 s horizon while cutting the worst autocorrelation. Raw
   100 ms rows would inflate sample counts 5× without adding information.
4. **Why is accuracy the wrong headline metric?** 73% of 1 s intervals have
   zero mid change, so "predict not-up" gets 84% accuracy. I report AUC and
   IC, and computed IC on non-overlapping subsamples to avoid overlap
   inflation.
5. **How realistic is the fill model?** Taker-only at displayed top-of-book
   after configurable latency — realistic for small size; no impact or
   sweep modeling. I deliberately did not simulate passive fills, because
   public data can't see queue position.
6. **What happens when latency increases?** Gross edge at 1 s goes from
   +0.57 bps (0 ms) to −0.20 bps (1,000 ms): the signal fully decays within
   ~1 s, consistent with the IC drop from 1 s to 5 s horizons.
7. **What is adverse selection here?** The maker filling my aggressive order
   loses ~0.5 bps in expectation when my signal fires. The signal's real use
   is for makers to skew/pull quotes when imbalance is against them.
8. **Why did ML add so little?** At 1–5 s, the book state is nearly the
   entire signal; nonlinear interactions with flow/vol features are
   second-order in a 2 h sample. More data might change that; it wasn't the
   research question.
9. **Microprice vs mid — what did you find?** Microprice − mid =
   (spread/2)·imbalance₁ exactly; with a 1-tick spread they're the same
   signal. On this instrument the question collapses; it becomes meaningful
   only when spreads vary.
10. **What would you test next?** Maker-side quoting simulation with
    conservative trade-through fills; multi-regime data; OFI vs snapshot
    imbalance; cross-venue fee structures that could flip the economics.

## LinkedIn post skeleton

> The most useful result in my latest project was a negative one.
>
> I captured ~4 hours of live Binance BTC/USDT order-book data and tested the
> oldest microstructure signal there is: book imbalance. It works — cleanly
> monotone decile plots, AUC 0.74, IC 0.18 out-of-sample with leakage-tested
> chronological validation.
>
> Then I made it trade. Buying the ask on strong signals earns +0.5 bps
> gross per trade — the signal even beats the spread, which on BTCUSDT is one
> tick 99.9% of the time. But at a 10 bps taker fee, every single trade loses.
> Breakeven fee: ~0.25 bps/side. That tier doesn't exist for takers.
>
> The information is real; it just belongs to market makers — as a reason to
> skew quotes, not to cross spreads. Sometimes the best thing a backtest can
> tell you is who's allowed to use a signal.
>
> Code, data pipeline, research note: [repo link]
