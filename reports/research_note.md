# Research note: Order-book information in BTC — who can monetize it?

**Dates:** v1 2026-08-25 · v2 capture 2026-08-27→29 · **Venues:** Binance spot + USDS-M perp, BTCUSDT

## Question and design

v1 established that top-of-book imbalance predicts BTC/USDT spot mid moves at
1–5s but that a ~0.5bps edge cannot pay 10bps/side taker fees. v2 asks the
harder questions: does the prediction *replicate* across days and regimes; is
it tradable on the perp (legitimate shorts, 5bps taker, 2bps maker); does the
information help a passive maker; and does venue fragmentation (spot vs perp)
itself carry signal?

Design guards, fixed in advance: a holdout consisting of the final 12 hours
of collection was time-stamped 27 hours before it existed, and every
threshold, margin, and maker parameter was frozen from walk-forward folds at
the 24h checkpoint. Signal quality was measured separately from strategy P&L
(per-6h-fold non-overlapping Spearman ICs, sign consistency, regime
conditioning). "No trade" was an admissible optimum for the execution rule.

## Data

~46h (spot) / ~44h (perp) of clean 100ms top-20 book snapshots and trades
(3.1M snapshots, 6.7M trades) on a shared local clock, sampled to 500ms
(333k/317k rows). Coverage spans Asia/EU/US sessions, a quiet overnight tape
and a violent burst (276k perp trades/15min). A 2.5h perp-feed outage
(fstream unreachable on 08-28) and v1-era sleep gaps are masked, not bridged.
Perp trade rate ran ~3x spot. Spread is one tick 99.9% of the time on both.

## Finding 1 — prediction replicates, everywhere

Level-1/5/20 imbalance carries per-fold OOS Spearman ICs of 0.36–0.39 (spot)
and 0.31–0.33 (perp) with the same sign in 8/8 development folds; OFI
(0.25–0.28) and 1–30s momentum (0.10–0.26) also replicate. The composite
ridge signal, evaluated ONCE on the pre-registered holdout, scored Spearman
IC 0.30 (spot@1s), 0.43 (spot@5s), 0.26/0.35 (perp) — positive in 13 of 13
hourly blocks, cross-hour t of 16–20. Whatever else is true, this
information is real and stable.

## Finding 2 — the perp is where price discovery happens

The perp book predicts *spot's* next 1–5s at IC 0.30–0.32 (7/7 folds;
holdout 0.26) — reliably stronger than spot's book predicts perp (0.23–0.24).
Non-overlapping 500ms return cross-correlations agree: perp→spot at 0.5s lag
is 0.157 vs 0.134 the other way, and the perp lead persists to ~2s while the
spot lead dies by 1.5s. The basis (perp−spot) sat at −4.1bps (σ 0.75bps)
all week — a negative-carry regime — mean-reverting with a ~5s half-life,
with convergence carried by the spot leg (basis→spot IC 0.16; basis→perp
only −0.035).

## Finding 3 — no taker configuration is viable, including on the perp

With the EV rule trading only when |predicted move| exceeds round-trip cost
plus a wf-selected margin, the chosen policy was NO TRADE for every venue and
horizon — at 5bps perp taker, 10bps spot, and every sensitivity tier down to
1.8bps/side. The zero-fee diagnostic shows why: even the 98th-percentile
signal predicts ~0.5bps and the very strictest thresholds reach only
~0.9–1.4bps gross per trade (16–335 trades), i.e. 2.5x short of the
*cheapest* round-trip tested. The frozen no-trade policy was confirmed on
the holdout (zero trades, zero P&L — correctly, since any trade would have
paid more in fees than the signal predicts in movement).

## Finding 4 — the maker experiment, and an honest failure

The hypothesis: even unmonetizable directionally, the signal should let a
maker avoid adverse selection — pull the ask when the book says "up", the bid
when it says "down" (a discrete reservation-price skew; a 1-tick spread
leaves no room to quote inside), with inventory caps.

Mechanically it worked; economically it did not. Under conservative
trade-through fills at the VIP0 2bps maker fee, symmetric quoting loses
~3.1–3.4bps per fill; the frozen skew (98th-pct pull threshold, cap 3)
halves fill count and total losses in every period — but the *per-fill*
improvement that appeared at the 24h checkpoint (−4.98→−3.10bps) vanished on
the full walk-forward region (−3.12 vs −3.17) and the holdout (−3.37 vs
−3.31), and 1s markouts were marginally worse. Under optimistic touch fills
markouts do improve (−0.035→−0.025bps) but per-fill P&L doesn't. The sober
reading: on a book quoting one tick wide, spread capture (~0.007bps) is
three orders of magnitude too small for the adverse selection (−0.5 to −1bps
markout) plus fee; the signal reduces *exposure*, not fill *quality*, and
cannot flip the sign at any fee we tested down to zero (through-model
−1.1bps/fill at zero fee). Passive viability requires what public data
cannot see: queue position, rebates, and sub-tick economics.

## Interpretation

Every result points the same direction. The order book broadcasts real,
strongly replicating short-horizon information; the perp broadcasts it
first; and the fee/latency structure prices that information at almost
exactly its worth — leaving nothing for takers at any tier, and for makers
only the option not to stand in front of it. The v1 conclusion generalizes:
this is not an inefficiency, it is infrastructure. The economically valuable
residues are defensive (quote pulling as risk reduction) and operational
(perp-informed execution timing on spot), not directional.

## Limitations

Two days, one exchange, one (negative) basis regime, no weekend; holdout
spans EU+US hours only. Fill models bracket but cannot pin reality — resting
size would alter the queue we simulate against. Local clock offset (~−25ms)
makes absolute latency approximate; cross-venue timing shares one clock and
is clean. Overlapping labels mitigated by non-overlapping sampling
throughout.

## Next experiment

Priced-to-test: (1) the maker sim at rebate-tier fees with explicit
queue-position priors; (2) perp-signal-timed execution of a spot parent
order, measured as implementation shortfall — the likeliest place this
information pays anyone who isn't already an HFT.
