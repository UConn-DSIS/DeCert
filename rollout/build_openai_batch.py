#!/usr/bin/env python3
"""Build OpenAI Batch API request files for GPT-4o judge rollouts.

Batch-API alternative to rollout.py (half the cost): one request per
(example, rep) with custom_id "<example_id>__rep<r>" and the same prompt,
sampling parameters and seeds as rollout.py. Submit the files with
tools/submit_openai_batch.py, then convert the results with
rollout/merge_openai_batch.py.

Example:
  python rollout/build_openai_batch.py --input /path/to/helpsteer3_12000.jsonl \
      --split-name train --n 3 --out-dir batches/train_rollout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.batch_score import write_sharded_requests
from decert.data import load_examples
from decert.prompts import build_judge_messages
from decert.rubrics import get_rubric

BASE_SEED = 20250815


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    examples = load_examples(args.input, args.split_name)
    if args.limit is not None:
        examples = examples[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = []
    for example in examples:
        messages = build_judge_messages(example, get_rubric(example.domain))
        for rep in range(args.n):
            requests.append(
                {
                    "custom_id": f"{example.example_id}__rep{rep}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": messages,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "seed": BASE_SEED + rep,
                    },
                }
            )

    paths = write_sharded_requests(out_dir, f"batch_{args.split_name}", requests)
    print(f"Wrote {len(requests)} requests to {len(paths)} file(s):")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
