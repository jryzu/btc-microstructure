"""Collect Binance Spot public market data over websocket.

Streams captured (combined stream endpoint):
  - <symbol>@depth20@100ms : top-20 partial order-book snapshots (~10/sec)
  - <symbol>@aggTrade      : aggregate trades

Raw events are written as gzipped JSONL, one file segment per rotation
interval, under data/raw/. Each line is the raw combined-stream message
wrapped with a local receive timestamp:

    {"recv_ts_ms": <local unix ms>, "stream": "...", "data": {...}}

Notes on timestamps:
  - aggTrade events carry exchange event time "E" and trade time "T" (ms).
  - Spot partial-depth snapshots carry NO exchange timestamp (only
    lastUpdateId), so the local receive time is the analysis clock for
    order-book state. This is a documented limitation.

Usage:
    python -m src.collect --symbol BTCUSDT --duration-minutes 120
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import signal
import sys
import time
from pathlib import Path

import websockets

WS_BASES = {
    "spot": "wss://stream.binance.com:9443/stream",
    "perp": "wss://fstream.binance.com/stream",   # USDS-M perpetual futures
}
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _segment_path(venue: str, symbol: str, t: float) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(t))
    return RAW_DIR / f"{venue}_{symbol}_{stamp}.jsonl.gz"


async def collect(symbol: str, duration_minutes: float, rotate_minutes: float = 15.0,
                  venue: str = "spot") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sym = symbol.lower()
    # USDS-M futures: the @aggTrade stream delivers no data on fstream
    # (verified empirically 2026-08-27); @trade works and carries p/q/m/E/T.
    trade_stream = "aggTrade" if venue == "spot" else "trade"
    url = f"{WS_BASES[venue]}?streams={sym}@depth20@100ms/{sym}@{trade_stream}"
    deadline = time.time() + duration_minutes * 60.0

    stop = asyncio.Event()

    def _handle_sig(*_a) -> None:
        stop.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _handle_sig)
        except NotImplementedError:
            pass

    n_msgs = 0
    n_depth = 0
    n_trades = 0
    backoff = 1.0

    fh = None
    seg_end = 0.0

    def _rotate(now: float):
        nonlocal fh, seg_end
        if fh is not None:
            fh.close()
        path = _segment_path(venue, symbol, now)
        fh = gzip.open(path, "at", encoding="utf-8")
        seg_end = now + rotate_minutes * 60.0
        print(f"[collect] writing {path.name}", flush=True)

    try:
        while time.time() < deadline and not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                    print(f"[collect] connected to {url}", flush=True)
                    backoff = 1.0
                    last_flush = time.time()
                    while time.time() < deadline and not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            continue
                        now = time.time()
                        if fh is None or now >= seg_end:
                            _rotate(now)
                        try:
                            obj = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        stream = obj.get("stream", "")
                        fh.write(json.dumps({
                            "recv_ts_ms": int(now * 1000),
                            "stream": stream,
                            "data": obj.get("data"),
                        }, separators=(",", ":")) + "\n")
                        n_msgs += 1
                        if "depth" in stream:
                            n_depth += 1
                        elif "aggTrade" in stream or "@trade" in stream:
                            n_trades += 1
                        if now - last_flush > 5.0:
                            fh.flush()
                            last_flush = now
                        if n_msgs % 5000 == 0:
                            remaining = (deadline - now) / 60.0
                            print(f"[collect] msgs={n_msgs} depth={n_depth} trades={n_trades} "
                                  f"remaining={remaining:.1f}min", flush=True)
            except (websockets.WebSocketException, OSError) as exc:
                if time.time() >= deadline or stop.is_set():
                    break
                print(f"[collect] connection error: {exc!r}; reconnecting in {backoff:.0f}s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
    finally:
        if fh is not None:
            fh.close()
        print(f"[collect] done. msgs={n_msgs} depth={n_depth} trades={n_trades}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Binance depth20 + aggTrade streams (spot or USDS-M perp).")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--venue", default="spot", choices=["spot", "perp"])
    ap.add_argument("--duration-minutes", type=float, default=120.0)
    ap.add_argument("--rotate-minutes", type=float, default=15.0)
    args = ap.parse_args()
    try:
        asyncio.run(collect(args.symbol, args.duration_minutes, args.rotate_minutes, args.venue))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
