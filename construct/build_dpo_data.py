#!/usr/bin/env python3
"""Build Max-Min DPO pairs from scored judgment groups.

Per group: chosen = argmax Q_total, rejected = argmin Q_total. Q_total is
outcome-dominant, so mixed groups give correct-vs-wrong pairs — kept
unconditionally, the verdict-correctness margin is the signal — while
all-correct groups give same-verdict certificate-quality pairs (nothing is
wasted); all-wrong groups are never sampled.

--same-verdict-delta (default 0.25) applies ONLY to
same-verdict pairs: an all-correct group whose process-score gap is at or
below the threshold is dropped, since a near-tied certificate pair is
verifier noise, not preference. To sweep delta cheaply, build the input pool
with --same-verdict-delta 0 and apply delta in merge_dpo_data.py.

--process-aggregation selects either the stored geometric Q_proc or the
arithmetic mean of S_snd, S_cov, and S_dec. All other construction rules are
unchanged.

Example:
  python construct/build_dpo_data.py --data /path/to/helpsteer3_12000.jsonl \
      --split-name train --scored scored/train.jsonl \
      --same-verdict-delta 0 --out training_data/train_dpo.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import load_examples, load_scored_groups, write_jsonl
from decert.pairs import PROCESS_AGGREGATIONS, build_maxmin_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--scored", nargs="+", required=True)
    parser.add_argument(
        "--same-verdict-delta",
        type=float,
        default=0.25,
        help="minimum selected process-score gap for SAME-VERDICT pairs "
        "(all-correct groups); mixed correct-vs-wrong pairs are exempt",
    )
    parser.add_argument(
        "--process-aggregation",
        choices=PROCESS_AGGREGATIONS,
        default="geometric",
        help="aggregation used for Max-Min ranking and same-verdict gaps",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.same_verdict_delta < 0.0:
        raise SystemExit("--same-verdict-delta must be non-negative")
    examples_by_id = {
        ex.example_id: ex for ex in load_examples(args.data, args.split_name)
    }

    groups, load_stats = load_scored_groups(args.scored, examples_by_id)

    records, stats = build_maxmin_pairs(
        groups,
        examples_by_id,
        same_verdict_delta=args.same_verdict_delta,
        process_aggregation=args.process_aggregation,
    )
    n = write_jsonl(args.out, records)

    both_correct = sum(1 for r in records if r["rejected_scores"]["outcome"] == 1)
    print(
        f"groups loaded: {len(groups)} "
        f"(score errors skipped: {load_stats['errors']}, "
        f"discarded skipped: {load_stats['discarded']})"
    )
    print(
        "maxmin pairs "
        f"(aggregation={args.process_aggregation}, "
        f"same_verdict_delta={args.same_verdict_delta}):"
    )
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print(
        f"  pair mix: {n - both_correct} correct-vs-wrong, "
        f"{both_correct} same-verdict (all-correct groups)"
    )
    print(f"wrote {n} Max-Min DPO pairs -> {args.out}")


if __name__ == "__main__":
    main()
