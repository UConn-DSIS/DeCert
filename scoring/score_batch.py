#!/usr/bin/env python3
"""Batch API scoring: build phase-1/phase-2 request files and assemble scores.

Flow (submission itself uses submit_openai_batch.py):

  python scoring/score_batch.py phase1 --data /path/to/helpsteer3_12000.jsonl \
      --split-name train \
      --rollouts rollouts/gpt4o_train.jsonl rollouts/qwen3_train.jsonl \
      --verifier-model gpt-4o --out-dir batches/score_train
  python tools/submit_openai_batch.py \
      --requests batches/score_train/phase1_*[0-9].jsonl

  python scoring/score_batch.py phase2 ... --verifier-model gpt-4o-mini \
      --phase1-output batches/score_train/phase1_*_output.jsonl
  python tools/submit_openai_batch.py \
      --requests batches/score_train/phase2_*[0-9].jsonl

  python scoring/score_batch.py assemble ... \
      --phase1-output ... --phase2-output ... --out scored/train.jsonl

Phase 2 contains only matcher requests, so passing a cheaper
--verifier-model there is the Batch-API equivalent of --matcher-model in
score_judgments.py.

IMPORTANT: pass the same --rollouts files in the same order to every phase —
judgment indices inside custom_ids depend on that order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.batch_score import (
    assemble_groups,
    phase1_requests,
    phase2_requests,
    read_batch_texts,
    required_phase1_ids,
    required_phase2_ids,
    write_sharded_requests,
)
from decert.data import load_examples, load_rollout_groups, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data", required=True, help="local HelpSteer3 JSONL")
        p.add_argument("--split-name", required=True)
        p.add_argument("--rollouts", nargs="+", required=True)

    p1 = sub.add_parser("phase1", help="analyzer + grounding + validity requests")
    common(p1)
    p1.add_argument(
        "--analyzer-model",
        default=None,
        help="analyzer model; default: same as --verifier-model",
    )
    p1.add_argument("--verifier-model", required=True)
    p1.add_argument("--out-dir", required=True)

    p2 = sub.add_parser("phase2", help="matcher requests (needs phase1 outputs)")
    common(p2)
    p2.add_argument("--verifier-model", required=True)
    p2.add_argument("--phase1-output", nargs="+", required=True)
    p2.add_argument("--out-dir", required=True)

    pa = sub.add_parser("assemble", help="compute scores locally (no API)")
    common(pa)
    pa.add_argument("--phase1-output", nargs="+", required=True)
    pa.add_argument("--phase2-output", nargs="+", required=True)
    pa.add_argument("--lam", type=float, default=0.5)
    pa.add_argument(
        "--local-validity-threshold",
        type=float,
        default=1.0,
        help="minimum support-chain validity required to constrain a "
        "criterion interval (default: 1.0)",
    )
    pa.add_argument("--out", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = {ex.example_id: ex for ex in load_examples(args.data, args.split_name)}
    groups = load_rollout_groups(args.rollouts)
    unknown_ids = sorted(set(groups) - set(examples))
    if unknown_ids:
        preview = ", ".join(unknown_ids[:5])
        raise SystemExit(
            f"{len(unknown_ids)} rollout example IDs are absent from --data "
            f"under split {args.split_name!r} ({preview}); check --split-name"
        )
    print(f"{len(groups)} rollout groups")

    if args.command == "phase1":
        requests = phase1_requests(
            examples,
            groups,
            args.verifier_model,
            analyzer_model=args.analyzer_model,
        )
        paths = write_sharded_requests(args.out_dir, "phase1", requests)
        print(f"wrote {len(requests)} phase1 requests to {len(paths)} file(s):")
        for p in paths:
            print(f"  {p}")

    elif args.command == "phase2":
        phase1_texts = read_batch_texts(args.phase1_output)
        print(f"loaded {len(phase1_texts)} phase1 results")
        missing = required_phase1_ids(examples, groups) - set(phase1_texts)
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise SystemExit(
                f"phase 1 is missing {len(missing)} requested results ({preview}); "
                "replace the incomplete batch result before phase 2"
            )
        requests = phase2_requests(examples, groups, phase1_texts, args.verifier_model)
        paths = write_sharded_requests(args.out_dir, "phase2", requests)
        print(f"wrote {len(requests)} phase2 requests to {len(paths)} file(s):")
        for p in paths:
            print(f"  {p}")

    elif args.command == "assemble":
        if not 0.0 < args.lam < 1.0:
            raise SystemExit("--lam must be in (0, 1)")
        if not 0.0 < args.local_validity_threshold <= 1.0:
            raise SystemExit("--local-validity-threshold must be in (0, 1]")
        phase1_texts = read_batch_texts(args.phase1_output)
        phase2_texts = read_batch_texts(args.phase2_output)
        print(f"loaded {len(phase1_texts)} phase1 / {len(phase2_texts)} phase2 results")
        missing_phase1 = required_phase1_ids(examples, groups) - set(phase1_texts)
        missing_phase2 = required_phase2_ids(examples, groups, phase1_texts) - set(
            phase2_texts
        )
        if missing_phase1 or missing_phase2:
            missing = sorted(missing_phase1 | missing_phase2)
            preview = ", ".join(missing[:5])
            raise SystemExit(
                f"batch scoring is missing {len(missing)} requested results "
                f"({preview}); replace the incomplete batch result before assembly"
            )
        records = assemble_groups(
            examples,
            groups,
            phase1_texts,
            phase2_texts,
            lam=args.lam,
            tau_loc=args.local_validity_threshold,
        )
        n = write_jsonl(args.out, records)
        n_parse = sum(j["parse_ok"] for r in records for j in r.get("judgments", []))
        n_total = sum(len(r.get("judgments", [])) for r in records)
        discarded = sum(1 for r in records if r.get("discarded"))
        print(f"wrote {n} group records to {args.out}")
        print(f"  judgments: {n_parse}/{n_total} parseable")
        print(f"  groups discarded for empty issue set: {discarded}")


if __name__ == "__main__":
    main()
