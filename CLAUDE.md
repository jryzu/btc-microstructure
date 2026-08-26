# Crypto Microstructure Research Lab

## Mission

Build a polished, public, recruiter-facing quantitative trading research project focused on BTC/USDT spot market microstructure.

This is NOT primarily a software-engineering exercise, Kaggle ML exercise, or attempt to fabricate a profitable strategy.

The purpose is to demonstrate that I can:

1. work with real high-frequency crypto market data;
2. formulate a sensible trading research question;
3. construct microstructure features;
4. avoid obvious time-series leakage;
5. evaluate predictive signals out-of-sample;
6. translate prediction into realistic executable economics;
7. reason about spread, fees, latency, inventory, fills, and adverse selection;
8. clearly communicate what worked, what failed, and why.

The final repository should be credible to a trader or quantitative researcher at a small crypto market maker such as Kairon Labs, Wintermute, or a similar crypto-native trading team.

Optimize for **research credibility and clarity**, not complexity.

---

# Time constraint

This is a rapid MVP.

Target: approximately one focused day of agentic work plus passive data collection.

Do NOT overengineer.

Prioritize in this order:

1. correct data;
2. defensible analysis;
3. economic interpretation;
4. polished README;
5. clean code;
6. optional sophistication.

If an optional feature risks preventing completion of the core project, skip it.

---

# Core research question

Primary question:

> Does BTC/USDT order-book state contain short-horizon information about subsequent mid-price movement, and does any apparent predictive edge survive realistic trading frictions?

Primary horizons:

- 1 second
- 5 seconds

Optional:

- 10 seconds

We care much more about whether the analysis is honest and economically meaningful than whether a model achieves impressive predictive performance.

A result showing that apparent prediction disappears after spread, fees, or latency is a valid and potentially interesting result.

Never fabricate performance.

---

# Market and data

## Instrument

Default:

- Binance Spot
- BTCUSDT

Do not expand to many symbols during the MVP.

ETHUSDT may be added only after the BTC pipeline works end-to-end.

## Authentication

Use PUBLIC market-data endpoints only.

Do not request API keys.
Do not place live orders.
Do not trade real money.

## Preferred data

Capture:

1. top-of-book / order-book depth;
2. aggregate trades.

For the MVP, prefer Binance's public partial order-book depth stream with the top 20 levels at approximately 100 ms, plus the BTCUSDT aggregate-trade stream.

Store raw events locally before transformation.

Suggested structure:

```text
data/
  raw/
  processed/
```

Raw data should be timestamped and saved in an efficient/recoverable format.

JSONL compressed is acceptable for raw events.

Processed analytical datasets should preferably be Parquet.

Keep both:

- exchange event timestamp;
- local receive timestamp when possible.

This allows discussion of latency and data-quality limitations.

---

# Data collection strategy

Start the data collector as early as possible so it can accumulate observations while the rest of the repository is being built.

Aim for at least 1–2 hours of live observations during development.

Prefer several hours if the environment/session permits.

The pipeline must still function on a smaller sample so development does not block on collection time.

Build the collector so that additional hours can easily be collected later with one command.

Create a reproducible command such as:

```bash
python -m src.collect --symbol BTCUSDT --duration-minutes 120
```

Do not hardcode a specific collection duration.

---

# Required market-state variables

At each analysis timestamp calculate, where available:

## Prices

- best bid
- best ask
- mid-price
- spread
- spread in basis points
- microprice

Microprice should use top-level bid/ask sizes, with the formula documented.

## Depth / imbalance

Calculate bid-vs-ask imbalance for several depths, e.g.:

- level 1
- top 5
- top 10
- top 20

A reasonable definition is:

\[
I_k =
\frac{Q^{bid}_k-Q^{ask}_k}
{Q^{bid}_k+Q^{ask}_k}
\]

where quantities are aggregated across the first \(k\) levels.

## Trade flow

Using aggregate trades, derive sensible rolling features such as:

- signed trade volume;
- buy/sell imbalance;
- trade count;
- volume;
- average trade size.

Potential windows:

- 1 second
- 5 seconds
- 30 seconds

Infer aggressor direction only when justified by the source fields and document the convention.

## Dynamics

Consider:

- recent mid-price returns;
- realized volatility;
- spread changes;
- depth changes;
- order-flow imbalance if it can be constructed reliably.

Do not create dozens of arbitrary technical indicators.

Keep feature engineering interpretable.

---

# Labels

Construct future mid-price returns:

\[
r_{t,h}
=
\log\left(\frac{mid_{t+h}}{mid_t}\right)
\]

for \(h=1s\) and \(5s\).

Also construct a directional target if useful:

\[
y_{t,h}=1[r_{t,h}>0]
\]

Be extremely careful about timestamp alignment.

No future information can enter features.

Write at least one test specifically designed to catch label leakage / incorrect future alignment.

---

# Sampling

Raw order-book events are highly autocorrelated.

Do not naïvely treat every 100 ms observation as an independent sample.

Use a sensible regular analysis grid such as:

- 250 ms;
- 500 ms;
- or 1 second.

Choose one after inspecting the data and explain the choice.

---

# Modeling

Do NOT make deep learning the centerpiece.

The project should demonstrate research judgment rather than model complexity.

Required models:

## Baselines

At minimum:

- unconditional / no-signal benchmark;
- simple imbalance heuristic;
- logistic regression for directional prediction OR linear regression for future returns.

## Nonlinear model

Add ONE simple nonlinear model if useful, such as:

- HistGradientBoosting;
- XGBoost/LightGBM if dependencies are reasonable.

Do not spend time tuning an enormous hyperparameter search.

The model should help answer the economic research question.

---

# Validation

Absolutely no random train/test split.

Use chronological evaluation.

Prefer:

- train;
- validation;
- final held-out test;

or walk-forward evaluation if it can be implemented cleanly.

Never optimize thresholds on the test set.

Report both predictive and economic metrics.

Predictive metrics may include:

- accuracy;
- ROC AUC;
- log loss;
- correlation / information coefficient;
- calibration where appropriate.

Do not headline accuracy if the class balance makes it misleading.

---

# Trading/economic evaluation

Prediction alone is insufficient.

Translate the signal into at least one simple execution experiment.

## Required: aggressive execution test

For sufficiently strong positive signals:

- buy at the ask after configurable execution latency;
- close after the chosen horizon at the bid.

For negative signals:

- sell at bid;
- close at ask.

Include configurable:

- latency;
- taker fee;
- signal threshold.

Evaluate:

- gross P&L;
- net P&L;
- turnover;
- trade count;
- average edge per trade;
- win rate;
- maximum drawdown if meaningful;
- sensitivity to fees;
- sensitivity to latency.

The purpose is explicitly to answer:

> Is the statistical signal large enough to overcome executable costs?

## Optional: simple market-making experiment

Only implement this after the core pipeline is complete.

A simple reservation-price approach is acceptable:

\[
reservation_t =
mid_t
+
\alpha \cdot signal_t
-
\gamma \cdot inventory_t
\]

Construct bid/ask quotes around the reservation price.

Any passive-fill model MUST clearly acknowledge that public market data does not reveal true queue position.

If simulated fills depend on trades touching/crossing the quote, document the assumption.

Prefer a conservative fill assumption and perform sensitivity analysis.

Do not report simulated passive P&L as though it were live-trading performance.

---

# Research analyses that matter

At minimum answer:

### 1. Does order-book imbalance predict subsequent mid-price direction?

Show a binned plot:

imbalance decile/bucket → average subsequent return.

### 2. Is microprice more informative than simple midpoint?

Test whether microprice-mid deviation predicts subsequent price movement.

### 3. Does the relationship change under different volatility regimes?

At least split the data into lower/higher recent-volatility observations.

### 4. Does prediction survive trading costs?

This should be one of the project's main conclusions.

### 5. How quickly does the signal decay?

Compare at least 1s vs 5s horizons.

---

# Required figures

Produce polished, readable figures saved under:

```text
reports/figures/
```

Target roughly 4–6 useful figures, not 25 mediocre ones.

Examples:

1. imbalance bucket vs future return;
2. predicted probability vs realized direction;
3. signal strength vs net P&L;
4. cumulative simulated P&L;
5. latency/fee sensitivity;
6. feature importance or coefficient plot.

Do not specify flashy styling.

Charts should be suitable for embedding in GitHub README.

---

# Repository structure

Prefer something approximately like:

```text
.
├── CLAUDE.md
├── README.md
├── PLAN.md
├── pyproject.toml
├── src/
│   ├── collect.py
│   ├── preprocess.py
│   ├── features.py
│   ├── labels.py
│   ├── models.py
│   ├── backtest.py
│   └── plots.py
├── tests/
├── notebooks/
│   └── research.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   ├── research_note.md
│   └── results.json
├── RUNBOOK.md
└── STATUS.md
```

Do not create files simply to match this structure if they provide no value.

---

# Reproducibility

Someone cloning the project should quickly understand how to:

1. install dependencies;
2. collect data;
3. process data;
4. run the analysis;
5. reproduce figures.

Create `RUNBOOK.md` with exact commands.

Prefer a lightweight Python environment.

Use the simplest reliable dependency setup available.

Do not introduce Docker unless clearly necessary.

---

# Testing

Write focused tests for things where errors would invalidate the research:

- bid/ask calculations;
- imbalance;
- microprice;
- timestamp ordering;
- future-label alignment;
- trading-cost arithmetic;
- inventory/P&L accounting if market-making is implemented.

Do not chase 100% coverage.

---

# README requirements

README is one of the most important deliverables.

A trader should understand the project in under two minutes.

Use this approximate structure:

# BTC Microstructure: Does Order-Book Imbalance Survive Execution Costs?

## TL;DR

Three to five bullets stating what was tested and the actual results.

## Research Question

Brief explanation.

## Data

Venue, instrument, period, sampling, number of observations.

## Features

Simple explanation of imbalance, microprice, order flow, volatility.

## Method

Chronological validation and models.

## Results

Include key plots.

## Trading Reality Check

Explicitly discuss:

- spread;
- fees;
- latency;
- fills;
- adverse selection.

## What I Learned

Show judgment.

## Limitations

Be candid.

## Next Experiments

Two to four intelligent next steps.

## Reproduce

Minimal commands.

Avoid inflated language like:

- "highly profitable";
- "production-grade HFT";
- "alpha-generating system";

unless evidence actually supports it.

---

# Research note

Create:

```text
reports/research_note.md
```

Target approximately 1,000–1,500 words.

Write like an internal junior quant/trader research memo.

Focus on:

- hypothesis;
- methodology;
- evidence;
- economic interpretation;
- limitations;
- next experiment.

Do not write an academic literature review.

---

# Career-facing outputs

At completion, create a section in `STATUS.md` containing:

## Resume bullet options

Produce 3 concise alternatives based ONLY on results actually achieved.

The best one should follow roughly:

> Built BTC/USDT market-microstructure research pipeline from live order-book/trade data; tested order-book imbalance and microprice signals using chronological validation and evaluated execution robustness under spread, fees and latency.

Replace/generalize this based on actual findings.

Never invent performance numbers.

## Interview talking points

Create 8–10 questions a crypto trader might ask about the project, with concise answer notes.

Examples:

- Why should imbalance predict price?
- What leakage risks existed?
- Why did you choose this sampling frequency?
- How realistic is your fill model?
- What happens when latency increases?
- How would this differ across exchanges?
- What is adverse selection here?
- What would you test next?

## LinkedIn post skeleton

Create a short, technically substantive post centered around ONE interesting finding.

Do not use "excited to announce."

---

# Engineering rules

- Python 3.11+ preferred.
- Use type hints where helpful.
- Keep modules small and readable.
- Prefer pandas/polars/numpy/scikit-learn/matplotlib over excessive frameworks.
- Do not use an LLM API.
- Do not use paid data.
- Never commit secrets.
- Never require exchange authentication.
- Never place live trades.
- Never claim simulated results are real trading results.
- Avoid unnecessary abstraction.

---

# Agent behavior

When making implementation choices:

1. prefer completion over sophistication;
2. make sensible assumptions without repeatedly asking the user;
3. document assumptions;
4. test research-critical logic;
5. inspect actual generated data rather than assuming schemas;
6. if an API has changed, check current official documentation;
7. do not silently substitute synthetic data for real market data;
8. if blocked, implement the rest of the pipeline and clearly document the blocker.

Maintain a concise `STATUS.md` throughout the project.

At the end, `STATUS.md` must state:

- what works;
- what does not;
- data collected;
- tests run;
- key results;
- major assumptions;
- known limitations;
- exact next actions.

The repository should finish in a state where I could confidently send its GitHub URL to a crypto trader.