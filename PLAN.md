# PLAN

## Goal

One-day MVP answering: *does BTC/USDT top-of-book state (imbalance, microprice) predict 1s/5s mid-price moves, and does any edge survive spread + taker fees + latency?*

## Environment facts (verified)

- macOS, no system package managers; installed `uv` + Python 3.11 venv at `.venv`.
- Binance Spot public REST (`api.binance.com`) reachable (HTTP 200, live BTCUSDT depth returned).
- Websocket combined stream: `wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade`.
- Spot partial-depth (`depth20@100ms`) payloads carry **no exchange timestamp** (only `lastUpdateId`); local receive time is the order-book clock. aggTrade carries exchange event/trade times. Documented as a limitation.

## Implementation sequence

1. **Collector** (`src/collect.py`) — websocket → gzipped JSONL in `data/raw/`, 15-min rotation, reconnect logic. *Start immediately; runs in background (~3–4h target, works with less).*
2. **Preprocess** (`src/preprocess.py`) — raw JSONL → two Parquet tables: `book` (recv_ts, 20 bid/ask levels) and `trades` (ts, price, qty, aggressor side from `m` flag). Sanity checks: monotone timestamps, crossed-book filter, gap detection.
3. **Features** (`src/features.py`) — on a regular grid (default 500 ms, chosen after inspecting inter-snapshot spacing): mid, spread(bps), microprice deviation, imbalance at k∈{1,5,10,20}, signed trade flow / trade counts over 1s/5s/30s windows, short-horizon returns, rolling realized vol.
4. **Labels** (`src/labels.py`) — forward log mid returns at h∈{1s,5s} via strictly-future asof alignment + directional labels.
5. **Tests** (`tests/`) — imbalance/microprice arithmetic, timestamp ordering, an explicit label-leakage test (a feature computed *from* the future must not be reachable; forward return at t must equal mid at t+h vs t), execution-cost arithmetic.
6. **Models** (`src/models.py`) — chronological 60/20/20 train/val/test split; baselines (unconditional, imbalance-sign heuristic, logistic regression); one HistGradientBoosting model. Metrics: AUC, log loss, IC (Spearman/Pearson of predicted vs realized return), calibration.
7. **Backtest** (`src/backtest.py`) — aggressive taker test: signal > threshold → buy ask (after latency), exit at bid after horizon; symmetric for shorts. Configurable latency, taker fee, threshold. Threshold picked on validation only. Gross/net P&L, trade count, avg edge/trade, win rate, fee & latency sensitivity grids.
8. **Figures** (`src/plots.py`) — ~5 figures into `reports/figures/`: imbalance decile vs forward return; microprice-deviation vs forward return; calibration/pred-vs-realized; net P&L vs threshold under fee/latency scenarios; cumulative P&L; signal decay 1s vs 5s (+ vol-regime split).
9. **Write-ups** — `reports/research_note.md` (~1,200 words), README (recruiter-facing), RUNBOOK, STATUS with resume bullets / interview Q&A / LinkedIn skeleton.
10. **Final pass** — rerun end-to-end on full collected sample, critical review, fix top weaknesses.

## Acceptance criteria

- `python -m src.collect --symbol BTCUSDT --duration-minutes N` collects real data; ≥1–2h captured during the session.
- `pytest` green, including a leakage-specific test.
- End-to-end rerun from raw data → figures with documented commands (RUNBOOK).
- Results honestly reported: predictive stats out-of-sample + explicit net-of-cost conclusion (an edge that dies after costs is a valid result).
- README readable by a trader in <2 minutes; no fabricated numbers.

## Risks / blockers

- **Session/network interruption during collection** → collector rotates files every 15 min and reconnects; pipeline works on partial data.
- **Short sample (hours, one regime)** → acknowledged in Limitations; not fixable in one day.
- **No exchange ts on depth snapshots** → use local recv clock; discuss latency implications.
- No other blockers identified: endpoints verified live, no auth needed, no paid data.
