#!/usr/bin/env python3
"""Build SFT targets from scored judgment groups.

Per group: target = the verdict-correct judgment with the highest Q_proc;
groups with no correct judgment are skipped. --min-qproc defaults to 0.0
(any correct judgment qualifies, as in the paper); a positive value adds a
certificate-quality floor on the target.

Example:
  python construct/build_sft_data.py --data /path/to/helpsteer3_12000.jsonl \
      --split-name train --scored scored/train.jsonl \
      --out training_data/train_sft.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import load_examples, load_scored_groups, write_jsonl
from decert.pairs import build_sft_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--scored", nargs="+", required=True)
    parser.add_argument(
        "--min-qproc",
        type=float,
        default=0.0,
        help="optional quality floor on the target certificate "
        "(0.0 = verdict correctness is the only requirement)",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_qproc <= 1.0:
        raise SystemExit("--min-qproc must be in [0, 1]")
    examples_by_id = {
        ex.example_id: ex for ex in load_examples(args.data, args.split_name)
    }

    groups, load_stats = load_scored_groups(args.scored, examples_by_id)

    records, stats = build_sft_targets(groups, examples_by_id, min_qproc=args.min_qproc)
    n = write_jsonl(args.out, records)

    print(
        f"groups loaded: {len(groups)} "
        f"(score errors skipped: {load_stats['errors']}, "
        f"discarded skipped: {load_stats['discarded']})"
    )
    print(f"sft targets (min_qproc={args.min_qproc}):")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print(f"wrote {n} SFT targets -> {args.out}")


if __name__ == "__main__":
    main()
