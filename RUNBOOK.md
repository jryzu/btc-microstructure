# RUNBOOK

Exact commands to reproduce everything from scratch.

## 1. Environment

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/) (what this project used):

```bash
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python websockets pandas numpy pyarrow scikit-learn matplotlib pytest
```

Or with plain pip:

```bash
python3.11 -m venv .venv
.venv/bin/pip install websockets pandas numpy pyarrow scikit-learn matplotlib pytest
```

All commands below run from the repository root. `PY=.venv/bin/python`.

## 2. Collect data (public endpoints, no API key)

```bash
.venv/bin/python -m src.collect --symbol BTCUSDT --venue spot --duration-minutes 2880
.venv/bin/python -m src.collect --symbol BTCUSDT --venue perp --duration-minutes 2880
```

- Spot: `depth20@100ms` + `aggTrade` from `wss://stream.binance.com:9443/stream`.
- Perp (USDS-M futures): `depth20@100ms` + `trade` from `wss://fstream.binance.com/stream`
  (the futures `@aggTrade` stream delivers no data; `@trade` is used instead).
- Writes gzipped JSONL segments to `data/raw/`, rotated every 15 minutes
  (`--rotate-minutes`), reconnecting automatically on drops.
- Safe to re-run; new segments accumulate alongside old ones and the whole
  directory is processed together.

## 3. Process raw data → Parquet

```bash
.venv/bin/python -m src.preprocess
```

Produces `data/processed/book_{spot,perp}.parquet`, `trades_{spot,perp}.parquet`,
and `preprocess_report.json` (drop counts, gaps, receive-latency stats).

## 4. Features and labels

```bash
.venv/bin/python -m src.features          # 500ms grid; builds both venues + cross-venue features
.venv/bin/python -m src.labels            # 1s and 5s horizons (--horizons-s)
```

Produces `features_{venue}.parquet` then `dataset_{venue}.parquet` (cross-venue
basis/lead-lag columns are added when both venues are present).

## 5. Signal quality (separate from trading)

```bash
.venv/bin/python -m src.signals --venue spot
.venv/bin/python -m src.signals --venue perp
```

Per-fold out-of-sample ICs, sign consistency, and regime/session conditioning
for every candidate signal, on the development region only. Writes
`reports/signals_{venue}.json`.

## 6. Models (walk-forward; final 25% holdout untouched)

```bash
.venv/bin/python -m src.models --venue spot
.venv/bin/python -m src.models --venue perp
```

Walk-forward out-of-sample predictions over the development region plus
holdout predictions (evaluated later, once). Writes
`predictions_{venue}_{1,5}s.parquet` and metrics into `reports/results.json`.

## 7. Cost-aware taker backtest (EV rule; "no trade" is a legal optimum)

```bash
.venv/bin/python -m src.backtest --venue perp            # selection on wf region
.venv/bin/python -m src.backtest --venue perp --final    # holdout, exactly once
```

## 8. Maker-side quoting simulation

```bash
.venv/bin/python -m src.maker --venue perp            # wf selection
.venv/bin/python -m src.maker --venue perp --final    # holdout, exactly once
```

Symmetric vs signal-skewed vs inventory-aware quoting, conservative
trade-through fills (touch fills as the optimistic bracket).

## 9. Figures

```bash
.venv/bin/python -m src.plots
```

Writes 6 PNGs to `reports/figures/`.

## 10. Tests

```bash
.venv/bin/python -m pytest -q
```

## One-shot rerun (after data exists)

```bash
.venv/bin/python -m src.preprocess && \
.venv/bin/python -m src.features && \
.venv/bin/python -m src.labels && \
.venv/bin/python -m src.signals --venue spot && .venv/bin/python -m src.signals --venue perp && \
.venv/bin/python -m src.models --venue spot && .venv/bin/python -m src.models --venue perp && \
.venv/bin/python -m src.backtest --venue perp && .venv/bin/python -m src.maker --venue perp && \
.venv/bin/python -m src.plots
```

(`--final` runs against the holdout are executed once, at the end of the study.)
