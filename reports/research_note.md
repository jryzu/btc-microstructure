# Research note: Does BTC/USDT order-book imbalance survive execution costs?

**Date:** 2026-08-25/26 · **Data:** Binance Spot BTCUSDT, live capture

## Hypothesis

Top-of-book state — the size imbalance between bids and asks — should contain
short-horizon information about mid-price direction: a queue that is much
heavier on the bid side implies buying pressure and near-term upward drift.
The economically relevant question is not whether this effect exists (it is
well documented across markets) but whether it is large enough, on this venue
and instrument, to overcome realistic execution costs for an aggressive
(taker) trader.

## Data

I captured Binance Spot public websocket streams for BTCUSDT on 2026-08-25,
18:46–22:55 UTC: partial order-book depth (top 20 levels, ~100 ms cadence,
80,587 snapshots) and aggregate trades (109,563 trades). After removing
capture gaps (the collection machine slept several times), the sample covers
~134 clean minutes. BTC traded between $78,178 and $79,237 — a quiet, mildly
drifting afternoon; results should be read as one regime, not a general claim.

Two data-quality facts shaped the design. First, spot partial-depth messages
carry no exchange timestamp, so the local receive clock is the analysis clock;
trades (which do carry exchange time) showed a median receive latency of
~64 ms with a p99 near 600 ms, and snapshot arrivals are bursty (mean spacing
exactly ~100 ms, but half arrive within 20 ms of the previous one). Second,
the BTCUSDT spread is almost degenerate: one tick ($0.01, ~0.0013 bps) in
99.9% of observations. That makes the spread nearly free to cross — an
important input to the conclusion.

Events were sampled onto a 500 ms grid (book state = last snapshot at or
before t, dropped if staler than 1 s). 500 ms halves the raw 100 ms
autocorrelation problem while leaving ≥2 observations per 1 s horizon;
15,975 grid rows resulted.

## Features and labels

Features at time t use only data received at or before t: book imbalance at
depths 1/5/10/20, microprice−mid deviation, spread, signed trade volume and
trade-count/imbalance over 1/5/30 s windows, past 1/5 s returns, 30 s realized
volatility, and 5 s spread/depth changes. Labels are forward log mid returns
at 1 s and 5 s, computed by exact timestamp-validated row shifts (a label is
NaN if the grid row exactly h ahead is missing). A dedicated test injects an
artificial future price jump and verifies it moves labels but no feature.

One algebraic point worth recording: microprice − mid = (spread/2) ×
level-1 imbalance, exactly. With the spread pinned at one tick, microprice
deviation and level-1 imbalance are the same signal up to scale, and the data
confirm their decile plots are indistinguishable. "Is microprice more
informative than mid?" therefore collapses into the imbalance question on
this instrument; microprice would only add information where the spread
varies.

## Predictive results (chronological 60/20/20 split, test = final ~27 min)

Imbalance predicts. The decile plot of level-1 imbalance against forward
return is cleanly monotone: roughly −0.34 bps to +0.30 bps average forward
1 s return from the most ask-heavy to the most bid-heavy decile (−0.83 to
+0.65 bps at 5 s). On the held-out test set:

- Raw imbalance alone: ROC AUC 0.74 (1 s), 0.67 (5 s) for direction.
- Logistic regression and gradient boosting add little over raw imbalance
  (AUC 0.75/0.68) — at these horizons the book is the signal, and the
  logistic coefficients are dominated by imbalance_1 and imbalance_5.
- A ridge regression predicting the return itself achieves an information
  coefficient of 0.19 at 1 s. Because 500 ms sampling overlaps 1–5 s labels,
  I recomputed the IC on non-overlapping subsamples: 0.18 at 1 s
  (n=1,540, p≈8e-13) and 0.12 at 5 s (n=305, p≈0.05). The signal is real
  and decays with horizon: per unit time the 1 s edge is far stronger.
- Directional accuracy is not a headline metric here: 73% of 1 s intervals
  have exactly zero mid change, so class balance makes accuracy misleading;
  AUC and IC are the honest measures.

Calibration of predicted P(up) is approximately diagonal at both horizons.
The relationship survives a volatility split: monotone in both the low- and
high-volatility halves, slightly steeper at the extremes in high volatility.

## Economic results: the signal does not survive taker fees

The execution experiment: when the ridge signal exceeds a threshold (chosen
on the validation set only, from |signal| quantiles), buy at the ask
prevailing after a configurable latency, exit at the bid after the horizon;
mirrored for shorts; one position at a time; taker fee charged both legs.

At the validation-chosen threshold (~98th percentile of |signal|), the test
set produced 47 trades at 1 s and 31 at 5 s. Average **gross** edge per trade
— after crossing the spread, before fees — was **+0.52 bps at 1 s**; at 5 s it was
already gone even gross (−0.05 bps). At 1 s the signal genuinely beats the spread. But at Binance's
standard 10 bps/side taker fee the average **net** result is **−19.5 bps per
trade**; every single trade lost net of fees. The fee sensitivity is linear
and brutal: the strategy breaks even at roughly **0.26 bps per side** at 1 s
— 40× below the standard fee tier and still below the best VIP taker tiers
(~1.6–2 bps). Latency matters second-order by comparison: raising execution
delay from 0 to 1,000 ms erodes the 1 s gross edge from +0.58 bps to −0.19
bps, i.e. the entire alpha is gone within about a second — consistent with
the 1 s-scale signal decay measured above.

Two caveats on the backtest itself. The test period drifted mildly downward
and the model went short in 45 of 47 trades, so the long side is essentially
untested. And fills are assumed at the displayed top-of-book quote with no
market impact — fine for small size, optimistic beyond it.

## Interpretation

The result is a textbook adverse-selection boundary. Order-book imbalance
contains real, statistically strong information at the 1-second scale, but
its magnitude (~0.5 bps conditional on a strong signal) is an order of
magnitude smaller than the taker fee. Anyone paying taker fees is on the
wrong side of the trade; the information can only be monetized by
participants whose marginal cost per trade is near zero or negative — i.e.
makers earning rebates, who are exactly the counterparties being adversely
selected by this signal. For a market maker the practical use of this result
is defensive and offensive at once: skew or pull quotes when imbalance is
against you; lean entries when it is with you.

## Limitations

~2.2 hours of usable data from one afternoon, one venue, one instrument, one
volatility regime; no exchange timestamps on depth snapshots (local clock);
capture gaps from machine sleep; partial-depth stream rather than a
reconstructed full book; overlapping labels partially mitigated by
non-overlap ICs; short-dominated test trades; no queue/fill modeling beyond
displayed quotes.

## Next experiment

The natural continuation is the maker side: a reservation-price quoting
simulation (quotes skewed by the imbalance signal and inventory) with a
conservative trade-through fill model, to test whether the signal's value
survives realistic queue assumptions. Secondary: longer multi-day capture
spanning volatility regimes; per-exchange fee-tier scenario analysis; and an
order-flow-imbalance (event-based) feature to compare against static book
imbalance.
