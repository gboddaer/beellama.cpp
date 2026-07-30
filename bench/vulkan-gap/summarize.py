#!/usr/bin/env python3
"""Summarize benchmark JSONL output: grouped stats and correctness checks."""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summary_stats(values):
    if not values:
        return None
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return {
        "median": med,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(deviations),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark JSONL records."
    )
    parser.add_argument("input", type=str, help="JSONL output file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"line {line_no}: bad JSON: {exc}", file=sys.stderr)

    # Group by (mode, prompt)
    groups = defaultdict(list)
    for rec in records:
        key = (rec.get("mode", ""), rec.get("prompt", ""))
        groups[key].append(rec)

    if not groups:
        print("no valid records found", file=sys.stderr)
        sys.exit(1)

    exit_code = 0

    for (mode, prompt), group in sorted(groups.items()):
        tps_values = [r["measurement"]["predicted_per_second"]
                      for r in group if r["measurement"]["predicted_per_second"] > 0]

        print(f"\n=== {mode} / {prompt} ({len(tps_values)} valid / {len(group)} total) ===")
        print(f"  revision:   {group[0].get('revision', '?')}")
        print(f"  server:     {group[0].get('server', '?')}")
        print(f"  server_sha: {group[0].get('server_sha256', '?')[:12]}...")
        print(f"  device:     {group[0].get('device', '?')}")

        if tps_values:
            stats = summary_stats(tps_values)
            print(f"  t/s:  median={stats['median']:.2f}  "
                  f"min={stats['min']:.2f}  "
                  f"max={stats['max']:.2f}  "
                  f"mad={stats['mad']:.2f}")

            # Acceptance stats
            accepts = [r["measurement"]["draft_accept_pct"]
                       for r in group if r["measurement"].get("draft_accept_pct") is not None]
            if accepts:
                a_stats = summary_stats(accepts)
                print(f"  accept: median={a_stats['median']:.1f}%  "
                      f"min={a_stats['min']:.1f}%  "
                      f"max={a_stats['max']:.1f}%  "
                      f"mad={a_stats['mad']:.1f}%")
        else:
            print("  t/s: no valid measurements")

        # Output hash set
        hashes = set(r["measurement"]["content_sha256"] for r in group)
        if len(hashes) > 1:
            print(f"  hashes: MULTIPLE ({len(hashes)}) - POSSIBLE INCORRECTNESS")
            for h in sorted(hashes):
                print(f"    {h}")
            exit_code = 1
        else:
            print(f"  hashes: single ({next(iter(hashes))[:16]}...)")

        # Minimum repetition check
        if len(tps_values) < 5:
            print(f"  WARNING: fewer than 5 repetitions "
                  f"(got {len(tps_values)}), cannot reliably compare")
            exit_code = 1

    if exit_code:
        print("\n=== SUMMARY: issues found ===", file=sys.stderr)
    else:
        print("\n=== SUMMARY: all checks passed ===")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
