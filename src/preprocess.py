"""Convert raw gzipped JSONL captures into analytical Parquet tables.

Outputs (data/processed/):
  book.parquet   : one row per depth20 snapshot
      recv_ts_ms, bid_px_0..19, bid_qty_0..19, ask_px_0..19, ask_qty_0..19,
      last_update_id
  trades.parquet : one row per aggregate trade
      event_ts_ms (exchange), trade_ts_ms (exchange), recv_ts_ms (local),
      price, qty, is_buyer_maker

Aggressor convention: for aggTrade, ``m`` (is_buyer_maker) True means the
buyer was the passive side, i.e. the trade was seller-initiated (aggressive
sell). Signed volume = +qty if aggressive buy (m == False), -qty otherwise.

Sanity filters applied and reported:
  - drops crossed/empty books (best_bid >= best_ask or missing levels);
  - drops rows with non-positive prices/quantities;
  - sorts by receive timestamp; reports out-of-order counts and gaps > 5s.

Usage:
    python -m src.preprocess [--raw-dir data/raw] [--out-dir data/processed]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_LEVELS = 20


def parse_raw_files(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    book_rows: list[dict] = []
    trade_rows: list[dict] = []
    files = sorted(raw_dir.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no raw capture files in {raw_dir}")
    for path in files:
        for obj in _read_segment(path):
            stream = obj.get("stream", "")
            data = obj.get("data") or {}
            recv = obj.get("recv_ts_ms")
            if recv is None:
                continue
            if "depth" in stream:
                bids = data.get("bids") or []
                asks = data.get("asks") or []
                if len(bids) < N_LEVELS or len(asks) < N_LEVELS:
                    continue
                row = {"recv_ts_ms": recv, "last_update_id": data.get("lastUpdateId")}
                for i in range(N_LEVELS):
                    row[f"bid_px_{i}"] = float(bids[i][0])
                    row[f"bid_qty_{i}"] = float(bids[i][1])
                    row[f"ask_px_{i}"] = float(asks[i][0])
                    row[f"ask_qty_{i}"] = float(asks[i][1])
                book_rows.append(row)
            elif "aggTrade" in stream:
                trade_rows.append({
                    "event_ts_ms": data.get("E"),
                    "trade_ts_ms": data.get("T"),
                    "recv_ts_ms": recv,
                    "price": float(data.get("p", "nan")),
                    "qty": float(data.get("q", "nan")),
                    "is_buyer_maker": bool(data.get("m")),
                })
    return pd.DataFrame(book_rows), pd.DataFrame(trade_rows)


def _read_segment(path: Path):
    """Yield parsed JSON objects, tolerating a truncated tail (a segment
    still being written by the collector, or one cut off by a crash)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (EOFError, OSError, gzip.BadGzipFile):
        return


def clean_book(book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_snapshots": len(book)}
    n0 = len(book)
    ooo = int((book["recv_ts_ms"].diff() < 0).sum())
    book = book.sort_values("recv_ts_ms", kind="stable").reset_index(drop=True)
    # de-duplicate identical lastUpdateId (reconnect overlap)
    book = book.drop_duplicates(subset=["last_update_id"], keep="first")
    crossed = book["bid_px_0"] >= book["ask_px_0"]
    nonpos = (book["bid_px_0"] <= 0) | (book["ask_px_0"] <= 0) | \
             (book["bid_qty_0"] <= 0) | (book["ask_qty_0"] <= 0)
    book = book[~(crossed | nonpos)].reset_index(drop=True)
    gaps = book["recv_ts_ms"].diff()
    stats.update({
        "out_of_order": ooo,
        "dropped_crossed": int(crossed.sum()),
        "dropped_nonpos": int(nonpos.sum()),
        "dropped_dupes": int(n0 - len(book) - crossed.sum() - nonpos.sum()),
        "gaps_over_5s": int((gaps > 5000).sum()),
        "max_gap_ms": int(gaps.max()) if len(book) > 1 else 0,
        "clean_snapshots": len(book),
        "median_spacing_ms": float(gaps.median()) if len(book) > 1 else float("nan"),
    })
    return book, stats


def clean_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_trades": len(trades)}
    trades = trades.dropna(subset=["event_ts_ms", "price", "qty"])
    trades = trades[(trades["price"] > 0) & (trades["qty"] > 0)]
    trades = trades.sort_values("event_ts_ms", kind="stable").reset_index(drop=True)
    if len(trades):
        lat = trades["recv_ts_ms"] - trades["event_ts_ms"]
        stats["median_recv_latency_ms"] = float(lat.median())
        stats["p99_recv_latency_ms"] = float(lat.quantile(0.99))
    stats["clean_trades"] = len(trades)
    return trades, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    book, trades = parse_raw_files(Path(args.raw_dir))
    book, bstats = clean_book(book)
    trades, tstats = clean_trades(trades)

    book.to_parquet(out_dir / "book.parquet", index=False)
    trades.to_parquet(out_dir / "trades.parquet", index=False)

    report = {"book": bstats, "trades": tstats}
    print(json.dumps(report, indent=2))
    with open(out_dir / "preprocess_report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
