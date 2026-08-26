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
.venv/bin/python -m src.collect --symbol BTCUSDT --duration-minutes 120
```

- Streams `btcusdt@depth20@100ms` (top-20 book snapshots) and `btcusdt@aggTrade`
  from `wss://stream.binance.com:9443/stream`.
- Writes gzipped JSONL segments to `data/raw/`, rotated every 15 minutes
  (`--rotate-minutes`), reconnecting automatically on drops.
- Safe to re-run; new segments accumulate alongside old ones and the whole
  directory is processed together.

## 3. Process raw data → Parquet

```bash
.venv/bin/python -m src.preprocess
```

Produces `data/processed/book.parquet`, `data/processed/trades.parquet`, and
`preprocess_report.json` (drop counts, gaps, receive-latency stats).

## 4. Features and labels

```bash
.venv/bin/python -m src.features          # 500ms grid by default (--grid-ms)
.venv/bin/python -m src.labels            # 1s and 5s horizons (--horizons-s)
```

Produces `data/processed/features.parquet` then `dataset.parquet`.

## 5. Models (chronological 60/20/20 split)

```bash
.venv/bin/python -m src.models
```

Prints test metrics, writes them into `reports/results.json`, and stores
per-row predictions in `data/processed/predictions_{1,5}s.parquet`.

## 6. Cost-aware execution backtest

```bash
.venv/bin/python -m src.backtest --model p_gbt --fee-bps 10 --latency-ms 100
```

Threshold is selected on the validation set only, then applied to the test
set; fee/latency sensitivity grids are computed on the test set. Results are
appended to `reports/results.json`.

## 7. Figures

```bash
.venv/bin/python -m src.plots
```

Writes 6 PNGs to `reports/figures/`.

## 8. Tests

```bash
.venv/bin/python -m pytest -q
```

## One-shot rerun (after data exists)

```bash
.venv/bin/python -m src.preprocess && \
.venv/bin/python -m src.features && \
.venv/bin/python -m src.labels && \
.venv/bin/python -m src.models && \
.venv/bin/python -m src.backtest && \
.venv/bin/python -m src.plots
```
