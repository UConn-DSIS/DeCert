#!/usr/bin/env python3
"""Sample N decision-certificate judgments per example via any
OpenAI-compatible endpoint (OpenAI API or a local vLLM server).

Rollout configuration used in the paper:
  GPT-4o (3 samples per example):
    python rollout/rollout.py --input /path/to/helpsteer3_12000.jsonl \
        --split-name train \
        --model gpt-4o --n 3 --temperature 1.0 \
        --out rollouts/gpt4o_train.jsonl
  Qwen3-8B (5 samples per example; start `vllm serve Qwen/Qwen3-8B` first):
    python rollout/rollout.py --input /path/to/helpsteer3_12000.jsonl \
        --split-name train \
        --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 \
        --api-key-env VLLM_API_KEY --disable-thinking --n 5 --temperature 1.3 \
        --out rollouts/qwen3_train.jsonl

Output: one line per example:
  {"example_id", "model", "generation": [texts...], "verdict": [A/B/None...]}
Appends and skips complete example_ids already present in --out. Failed or
incomplete records are removed and retried when the command is run again.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import load_examples, prepare_resume_file
from decert.llm import OpenAICompatClient
from decert.parsing import parse_judgment
from decert.prompts import build_judge_messages
from decert.rubrics import get_rubric

BASE_SEED = 20250815


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local HelpSteer3 JSONL")
    parser.add_argument("--split-name", required=True, help="e.g. train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--n", type=int, required=True, help="samples per example")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="vLLM reasoning models (Qwen3): suppress the <think> block via "
        "chat_template_kwargs so max_tokens goes to the judgment itself",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="reasoning effort for gpt-oss models",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap examples")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    examples = load_examples(args.input, args.split_name)
    if args.limit is not None:
        examples = examples[: args.limit]

    out_path = Path(args.out)
    sampling_info = {
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
        "reasoning_effort": args.reasoning_effort,
    }

    def is_complete(record: dict) -> bool:
        generations = record.get("generation")
        return (
            record.get("model") == args.model
            and record.get("sampling") == sampling_info
            and isinstance(generations, list)
            and len(generations) == args.n
            and all(
                isinstance(text, str) and not text.lstrip().startswith("<error>")
                for text in generations
            )
        )

    done = prepare_resume_file(out_path, is_complete)
    todo = [ex for ex in examples if ex.example_id not in done]
    print(f"{len(examples)} examples, {len(done)} already done, {len(todo)} to run")

    client = OpenAICompatClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        disable_thinking=args.disable_thinking,
        reasoning_effort=args.reasoning_effort,
    )

    def run_example(example):
        messages = build_judge_messages(example, get_rubric(example.domain))
        generations = []
        for rep in range(args.n):
            try:
                text = client.complete(
                    messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=BASE_SEED + rep,
                )
            except Exception as exc:  # noqa: BLE001 - retry on the next run
                text = f"<error>{exc}</error>"
            generations.append(text)
        return {
            "example_id": example.example_id,
            "model": args.model,
            "sampling": sampling_info,
            "generation": generations,
            "verdict": [parse_judgment(g).verdict for g in generations],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with (
        open(out_path, "a", encoding="utf-8") as f,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for record in pool.map(run_example, todo):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            wrote += 1
            if wrote % 20 == 0:
                print(f"  {wrote}/{len(todo)}")
    print(f"Done. Wrote {wrote} records to {out_path}")


if __name__ == "__main__":
    main()
