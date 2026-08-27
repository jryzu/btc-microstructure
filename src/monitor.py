"""Lightweight collection health check.

Exits non-zero (and prints WHY) if any venue looks unhealthy:
  - newest raw file older than --max-age-s (collector stalled/dead);
  - depth message rate in the newest segment implausibly low;
  - disk free below --min-free-gb.

Clock note: recv_ts_ms - E mixes true network latency with local clock
offset (observed ~-25 ms on 2026-08-27, i.e. local clock behind exchange).
Cross-venue comparisons are unaffected: both venues share the local clock.

Usage:  python -m src.monitor
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def check_venue(venue: str, max_age_s: float, min_depth_rate: float) -> list[str]:
    problems = []
    files = sorted(RAW.glob(f"{venue}_*.jsonl.gz"))
    if not files:
        return [f"{venue}: no capture files at all"]
    latest = files[-1]
    age = time.time() - latest.stat().st_mtime
    if age > max_age_s:
        problems.append(f"{venue}: newest file {latest.name} is {age:.0f}s old (> {max_age_s:.0f}s)")
    c: collections.Counter = collections.Counter()
    tmin = tmax = None
    try:
        with gzip.open(latest, "rt") as fh:
            for line in fh:
                o = json.loads(line)
                c[o["stream"]] += 1
                r = o["recv_ts_ms"]
                tmin = r if tmin is None else min(tmin, r)
                tmax = r if tmax is None else max(tmax, r)
    except (EOFError, OSError, json.JSONDecodeError):
        pass  # in-progress segment
    span = (tmax - tmin) / 1000 if tmin and tmax and tmax > tmin else 0
    depth = sum(v for k, v in c.items() if "depth" in k)
    trades = sum(v for k, v in c.items() if "aggTrade" in k or "@trade" in k)
    if span > 60:
        rate = depth / span
        if rate < min_depth_rate:
            problems.append(f"{venue}: depth rate {rate:.1f}/s < {min_depth_rate}/s")
        if trades == 0:
            problems.append(f"{venue}: ZERO trade messages in current segment")
    print(f"[monitor] {venue}: age={age:.0f}s span={span:.0f}s "
          f"depth={depth} trades={trades} files={len(files)}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-s", type=float, default=120.0)
    ap.add_argument("--min-depth-rate", type=float, default=2.0)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()

    problems = []
    for venue in ("spot", "perp"):
        problems += check_venue(venue, args.max_age_s, args.min_depth_rate)
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"[monitor] disk free: {free_gb:.0f} GB")
    if free_gb < args.min_free_gb:
        problems.append(f"disk free {free_gb:.1f} GB < {args.min_free_gb} GB")

    if problems:
        for p in problems:
            print("PROBLEM:", p)
        sys.exit(1)
    print("[monitor] all healthy")


if __name__ == "__main__":
    main()
