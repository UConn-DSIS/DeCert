#!/usr/bin/env python3
"""Merge Max-Min DPO files and split them by pair kind.

Mixed pairs (rejected verdict wrong) and same-verdict pairs (all-correct
groups, rejected verdict also correct) go to separate outputs, so the
outcome-only variant (mixed pairs only) can be trained from the same run;
--out-all additionally writes their union.

--same-verdict-delta is applied HERE at merge time so the threshold can be
swept locally without touching the scored data: build the input pools
with `build_dpo_data.py --same-verdict-delta 0` (the full pool) and rerun
this merge per delta. Careful: pools built with a larger delta silently cap
the sweep — keep pools at 0. The gap uses the process aggregation recorded
by build_dpo_data.py (files without that field are treated as geometric).

Example:
  python construct/merge_dpo_data.py \
      --inputs training_data/train_dpo.jsonl \
      --same-verdict-delta 0.25 \
      --out-mixed training_data/dpo_mixed.jsonl \
      --out-same-verdict training_data/dpo_same_verdict_d025.jsonl \
      --out-all training_data/dpo_all_d025.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import read_jsonl, write_jsonl


def process_aggregation(record: dict) -> str:
    return record.get("process_aggregation", "geometric")


def process_score(scores: dict) -> float:
    return scores.get("process_score", scores["q_proc"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Max-Min DPO files (build with "
                        "--same-verdict-delta 0 to keep the sweep range full)")
    parser.add_argument(
        "--same-verdict-delta",
        type=float,
        default=0.25,
        help="minimum selected process-score gap for same-verdict pairs, "
        "applied at merge time; mixed pairs are exempt",
    )
    parser.add_argument("--out-mixed", required=True)
    parser.add_argument("--out-same-verdict", required=True)
    parser.add_argument(
        "--out-all",
        default=None,
        help="optional combined output containing mixed then same-verdict pairs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.same_verdict_delta < 0.0:
        raise SystemExit("--same-verdict-delta must be non-negative")
    mixed, same, dropped = [], [], 0
    aggregations = set()
    for path in args.inputs:
        for rec in read_jsonl(path):
            aggregations.add(process_aggregation(rec))
            if rec["rejected_scores"]["outcome"] == 0:
                mixed.append(rec)
                continue
            gap = process_score(rec["chosen_scores"]) - process_score(
                rec["rejected_scores"]
            )
            if gap > args.same_verdict_delta:
                same.append(rec)
            else:
                dropped += 1

    if len(aggregations) > 1:
        raise SystemExit(
            "--inputs mix process aggregations: " + ", ".join(sorted(aggregations))
        )

    n_mixed = write_jsonl(args.out_mixed, mixed)
    n_same = write_jsonl(args.out_same_verdict, same)
    n_all = write_jsonl(args.out_all, [*mixed, *same]) if args.out_all else None
    print(f"inputs: {len(args.inputs)} files")
    print(f"process aggregation: {next(iter(aggregations), 'unknown')}")
    print(f"mixed pairs: {n_mixed} -> {args.out_mixed}")
    print(f"same-verdict pairs (delta={args.same_verdict_delta}): {n_same} "
          f"kept, {dropped} below delta -> {args.out_same_verdict}")
    if n_all is not None:
        print(f"combined pairs: {n_all} -> {args.out_all}")


if __name__ == "__main__":
    main()
