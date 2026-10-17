#!/usr/bin/env python3
"""Merge OpenAI Batch API output files into the rollout jsonl format
(one line per example with a "generation" list), matching rollout.py output.

Example:
  python rollout/merge_openai_batch.py \
      --batch-output batches/train_rollout/*_output.jsonl \
      --expected-n 3 --expected-examples 12000 \
      --out rollouts/gpt4o_train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import read_jsonl, write_jsonl
from decert.parsing import parse_judgment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-output", nargs="+", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--expected-n",
        type=int,
        default=3,
        help="required successful repetitions per example (default: 3)",
    )
    parser.add_argument(
        "--expected-examples",
        type=int,
        default=None,
        help="optional number of example IDs expected across all result files",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_n < 1:
        raise SystemExit("--expected-n must be at least 1")
    if args.expected_examples is not None and args.expected_examples < 1:
        raise SystemExit("--expected-examples must be at least 1")

    per_example: dict[str, dict[int, str]] = defaultdict(dict)
    seen_slots: set[tuple[str, int]] = set()
    seen_examples: set[str] = set()
    n_ok = 0
    n_error = 0
    n_no_response = 0
    for path in args.batch_output:
        for rec in read_jsonl(path):
            custom_id = rec.get("custom_id", "")
            if "__rep" not in custom_id:
                continue
            example_id, rep_text = custom_id.rsplit("__rep", 1)
            try:
                rep = int(rep_text)
            except ValueError as exc:
                raise SystemExit(f"invalid rollout custom_id: {custom_id!r}") from exc
            slot = (example_id, rep)
            if slot in seen_slots:
                raise SystemExit(f"duplicate batch result for {custom_id!r}")
            seen_slots.add(slot)
            seen_examples.add(example_id)
            if rec.get("response") is None:
                # A record without "response" is a REQUEST line, not a result.
                n_no_response += 1
                continue
            if rec.get("error"):
                n_error += 1
                continue
            body = rec["response"].get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                n_error += 1
                continue
            text = choices[0]["message"]["content"] or ""
            per_example[example_id][rep] = text
            n_ok += 1

    if n_no_response and not n_ok:
        raise SystemExit(
            f"All {n_no_response} records lack a 'response' field — this looks "
            "like a batch REQUEST file (build_openai_batch.py output). Pass the "
            "RESULT file downloaded from the completed batch (output_file_id), "
            "e.g. via tools/submit_openai_batch.py."
        )
    print(f"records: {n_ok} ok, {n_error} errored, {n_no_response} without response")

    if args.expected_examples is not None and len(seen_examples) != args.expected_examples:
        raise SystemExit(
            f"expected {args.expected_examples} examples, found {len(seen_examples)}; "
            "check that every batch output shard was supplied"
        )

    expected_reps = set(range(args.expected_n))
    incomplete = {}
    for example_id in seen_examples:
        actual_reps = set(per_example.get(example_id, {}))
        if actual_reps != expected_reps:
            incomplete[example_id] = {
                "missing": sorted(expected_reps - actual_reps),
                "unexpected": sorted(actual_reps - expected_reps),
            }
    if incomplete:
        preview = ", ".join(
            f"{example_id} missing {reps['missing']}, "
            f"unexpected {reps['unexpected']}"
            for example_id, reps in sorted(incomplete.items())[:5]
        )
        raise SystemExit(
            f"{len(incomplete)} examples do not have {args.expected_n} successful "
            f"repetitions ({preview}); rerun the affected request shard and replace "
            "its result file before merging"
        )

    records = []
    for example_id in sorted(per_example):
        reps = per_example[example_id]
        generations = [reps[r] for r in sorted(reps)]
        records.append(
            {
                "example_id": example_id,
                "model": args.model,
                "generation": generations,
                "verdict": [parse_judgment(g).verdict for g in generations],
            }
        )
    n = write_jsonl(args.out, records)
    print(f"Wrote {n} example records to {args.out}")


if __name__ == "__main__":
    main()
