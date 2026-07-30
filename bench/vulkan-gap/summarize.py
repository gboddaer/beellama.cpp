#!/usr/bin/env python3
"""Summarize benchmark JSONL output: grouped stats and correctness checks."""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summary_stats(values):
    """Calculate summary statistics."""
    if not values:
        return None
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return {
        "median": med,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(deviations),
        "count": len(values),
    }


def main():
    """Main entry point for summarizer."""
    parser = argparse.ArgumentParser(
        description="Summarize benchmark JSONL records."
    )
    parser.add_argument("input", type=str, help="JSONL output file.")
    parser.add_argument(
        "--allow-mixed-provenance", action="store_true",
        help="Allow combining different executables in one group.",
    )
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
        # Separate valid and invalid records
        valid_recs = [r for r in group if r.get("measurement", {}).get("valid", True)]
        invalid_recs = [r for r in group if not r.get("measurement", {}).get("valid", True)]

        print(f"\n=== {mode} / {prompt} ({len(valid_recs)} valid / {len(group)} total) ===")
        print(f"  revision:   {group[0].get('provenance', {}).get('source_head', '?')}")
        print(f"  server:     {group[0].get('provenance', {}).get('server_path', '?')}")
        print(f"  server_sha: {group[0].get('provenance', {}).get('server_sha256', '?')[:12]}...")
        print(f"  device:     {group[0].get('provenance', {}).get('device', '?')}")

        # Invalid records summary
        if invalid_recs:
            print(f"\n  INVALID RECORDS: {len(invalid_recs)}")
            invalid_reasons = defaultdict(int)
            for rec in invalid_recs:
                for reason in rec.get("measurement", {}).get("invalid_reasons", ["unknown"]):
                    invalid_reasons[reason] += 1
            for reason, count in sorted(invalid_reasons.items()):
                print(f"    - {reason}: {count}")
            exit_code = 1

        # Throughput stats (only from valid records)
        tps_values = [r["measurement"]["predicted_per_second"]
                      for r in valid_recs if r["measurement"]["predicted_per_second"] > 0]

        if tps_values:
            stats = summary_stats(tps_values)
            print(f"  t/s:  median={stats['median']:.2f}  "
                  f"min={stats['min']:.2f}  "
                  f"max={stats['max']:.2f}  "
                  f"mad={stats['mad']:.2f}  "
                  f"count={stats['count']}")

            # Acceptance stats
            accepts = [r["measurement"]["draft_accept_pct"]
                       for r in valid_recs if r["measurement"].get("draft_accept_pct") is not None]
            if accepts:
                a_stats = summary_stats(accepts)
                print(f"  accept: median={a_stats['median']:.1f}%  "
                      f"min={a_stats['min']:.1f}%  "
                      f"max={a_stats['max']:.1f}%  "
                      f"mad={a_stats['mad']:.1f}%")
        else:
            print("  t/s: no valid measurements")

        # Finish reason counts
        finish_reasons = defaultdict(int)
        for rec in group:
            fr = rec.get("measurement", {}).get("finish_reason", "unknown")
            finish_reasons[fr] += 1
        if finish_reasons:
            print(f"  finish_reason: {dict(finish_reasons)}")

        # Draft token counts
        draft_n_values = [r["measurement"]["draft_n"]
                          for r in group if "draft_n" in r.get("measurement", {})]
        if draft_n_values:
            print(f"  draft_n: {summary_stats(draft_n_values)}")

        # Output hash set
        hashes = set(r["measurement"]["content_sha256"] for r in valid_recs)
        if len(hashes) > 1:
            print(f"  hashes: MULTIPLE ({len(hashes)}) - POSSIBLE INCORRECTNESS")
            for h in sorted(hashes):
                print(f"    {h}")
            exit_code = 1
        elif hashes:
            print(f"  hashes: single ({next(iter(hashes))[:16]}...)")

        # Minimum repetition check
        if len(tps_values) < 5:
            print(f"  WARNING: fewer than 5 valid repetitions "
                  f"(got {len(tps_values)}), cannot reliably compare")
            exit_code = 1

        # Provenance check
        server_shas = set(r.get("provenance", {}).get("server_sha256", "")
                         for r in group)
        if len(server_shas) > 1 and not args.allow_mixed_provenance:
            print(f"  ERROR: mixed server SHAs detected, use --allow-mixed-provenance to override")
            exit_code = 1

    if exit_code:
        print("\n=== SUMMARY: issues found ===", file=sys.stderr)
    else:
        print("\n=== SUMMARY: all checks passed ===")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
