#!/usr/bin/env python3
"""
Parse SB_DIAGNOSTICS output from logs/perf/* and summarise FPS / frame time.

Usage:
    scripts/parse_perf_logs.py logs/perf/default_bevy.log [--csv summary.csv]

The script looks for lines emitted by the Bevy diagnostics logger:
    [Bevy diag] frame  12345 • fps  144.2 • frame   6.94 ms • app   2.31 ms

Stats reported:
    samples, mean, median (p50), p95, p99 for FPS / frame_ms / app_ms
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DIAG_RE = re.compile(
    r"frame\s+(?P<frame>\d+)\s+•\s+fps\s+(?P<fps>[0-9.]+)\s+•\s+frame\s+(?P<frame_ms>[0-9.]+)\s+ms(?:\s+•\s+app\s+(?P<app_ms>[0-9.]+)\s+ms)?"
)


@dataclass
class Sample:
    frame: int
    fps: float
    frame_ms: float
    app_ms: Optional[float]


@dataclass
class Summary:
    name: str
    samples: int
    fps_mean: float
    fps_p50: float
    fps_p95: float
    fps_p99: float
    frame_ms_mean: float
    frame_ms_p50: float
    frame_ms_p95: float
    frame_ms_p99: float
    app_ms_mean: Optional[float] = None
    app_ms_p95: Optional[float] = None


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_log(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            clean = strip_ansi(raw).strip()
            match = DIAG_RE.search(clean)
            if not match:
                continue
            frame = int(match.group("frame"))
            fps = float(match.group("fps"))
            frame_ms = float(match.group("frame_ms"))
            app_ms_str = match.group("app_ms")
            app_ms = float(app_ms_str) if app_ms_str else None
            samples.append(Sample(frame=frame, fps=fps, frame_ms=frame_ms, app_ms=app_ms))
    return samples


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def summarise(name: str, samples: List[Sample]) -> Summary:
    if not samples:
        raise ValueError(f"No diagnostics found in {name}")

    fps_values = [s.fps for s in samples]
    frame_ms_values = [s.frame_ms for s in samples]
    app_ms_values = [s.app_ms for s in samples if s.app_ms is not None]

    summary = Summary(
        name=name,
        samples=len(samples),
        fps_mean=statistics.mean(fps_values),
        fps_p50=statistics.median(fps_values),
        fps_p95=percentile(fps_values, 0.95),
        fps_p99=percentile(fps_values, 0.99),
        frame_ms_mean=statistics.mean(frame_ms_values),
        frame_ms_p50=statistics.median(frame_ms_values),
        frame_ms_p95=percentile(frame_ms_values, 0.95),
        frame_ms_p99=percentile(frame_ms_values, 0.99),
    )

    if app_ms_values:
        summary.app_ms_mean = statistics.mean(app_ms_values)
        summary.app_ms_p95 = percentile(app_ms_values, 0.95)

    return summary


def write_csv(path: Path, summaries: Iterable[Summary]) -> None:
    fieldnames = [
        "name",
        "samples",
        "fps_mean",
        "fps_p50",
        "fps_p95",
        "fps_p99",
        "frame_ms_mean",
        "frame_ms_p50",
        "frame_ms_p95",
        "frame_ms_p99",
        "app_ms_mean",
        "app_ms_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "name": summary.name,
                "samples": summary.samples,
                "fps_mean": f"{summary.fps_mean:.3f}",
                "fps_p50": f"{summary.fps_p50:.3f}",
                "fps_p95": f"{summary.fps_p95:.3f}",
                "fps_p99": f"{summary.fps_p99:.3f}",
                "frame_ms_mean": f"{summary.frame_ms_mean:.3f}",
                "frame_ms_p50": f"{summary.frame_ms_p50:.3f}",
                "frame_ms_p95": f"{summary.frame_ms_p95:.3f}",
                "frame_ms_p99": f"{summary.frame_ms_p99:.3f}",
                "app_ms_mean": "" if summary.app_ms_mean is None else f"{summary.app_ms_mean:.3f}",
                "app_ms_p95": "" if summary.app_ms_p95 is None else f"{summary.app_ms_p95:.3f}",
            })


def print_summary(summary: Summary) -> None:
    print(f"== {summary.name} ==")
    print(f"samples : {summary.samples}")
    print(f"fps     : mean {summary.fps_mean:.2f} | p50 {summary.fps_p50:.2f} | p95 {summary.fps_p95:.2f} | p99 {summary.fps_p99:.2f}")
    print(f"framems : mean {summary.frame_ms_mean:.3f} | p50 {summary.frame_ms_p50:.3f} | p95 {summary.frame_ms_p95:.3f} | p99 {summary.frame_ms_p99:.3f}")
    if summary.app_ms_mean is not None:
        print(f"app ms  : mean {summary.app_ms_mean:.3f} | p95 {summary.app_ms_p95:.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ScriptBots perf logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Log files containing SB_DIAGNOSTICS output.")
    parser.add_argument("--csv", type=Path, help="Optional path to write CSV summary.")
    args = parser.parse_args()

    summaries: List[Summary] = []
    for log_path in args.logs:
        samples = parse_log(log_path)
        if not samples:
            print(f"[warn] No diagnostics found in {log_path}")
            continue
        summary = summarise(log_path.name, samples)
        summaries.append(summary)
        print_summary(summary)

    if args.csv and summaries:
        write_csv(args.csv, summaries)
        print(f"Wrote CSV summary to {args.csv}")


if __name__ == "__main__":
    main()
