#!/usr/bin/env python3
"""Score rollout judgments with the decision-certificate reward.

Merges generations for each example across one or more rollout files (e.g.
3 gpt-4o + 5 qwen3-8b = one group of 8), then runs the Scorer: per example
one analyzer call, per judgment grounding + validity + matcher calls.

Example:
  python scoring/score_judgments.py \
      --data /path/to/helpsteer3_12000.jsonl --split-name train \
      --rollouts rollouts/gpt4o_train.jsonl rollouts/qwen3_train.jsonl \
      --out scored/train.jsonl

Use the same --data/--split-name pair as the rollout stage: example_ids are
"<split-name>:<line>" and must line up.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decert.data import load_examples, load_rollout_groups, prepare_resume_file
from decert.llm import OpenAICompatClient
from decert.pipeline import Scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="local HelpSteer3 JSONL")
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--analyzer-model", default="gpt-4o")
    parser.add_argument("--analyzer-base-url", default=None)
    parser.add_argument("--analyzer-api-key-env", default=None)
    parser.add_argument(
        "--analyzer-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override automatic thinking control for a reasoning analyzer",
    )
    parser.add_argument(
        "--analyzer-reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--verifier-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--verifier-base-url", "--base-url", default=None)
    parser.add_argument("--verifier-api-key-env", "--api-key-env", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="for gpt-oss verifier models on vLLM: cap reasoning effort so "
        "the token budget reaches the final JSON (recommend: low)",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override automatic thinking control for grounding and validity",
    )
    parser.add_argument(
        "--matcher-model",
        default="Qwen/Qwen3-14B",
        help="model for the matcher role (default: Qwen/Qwen3-14B)",
    )
    parser.add_argument("--matcher-base-url", default=None)
    parser.add_argument("--matcher-api-key-env", default=None)
    parser.add_argument(
        "--matcher-reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="reasoning effort for a gpt-oss matcher served by vLLM",
    )
    parser.add_argument(
        "--matcher-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override automatic thinking control for the matcher",
    )
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument(
        "--local-validity-threshold",
        type=float,
        default=1.0,
        help="minimum support-chain validity required to constrain a "
        "criterion interval (default: 1.0)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _is_qwen(model: str) -> bool:
    return "qwen" in model.lower()


def _resolved_endpoint(model: str, base_url: str | None) -> str | None:
    if base_url is not None:
        return base_url
    return "http://localhost:8000/v1" if _is_qwen(model) else None


def _resolved_key_env(base_url: str | None, key_env: str | None) -> str:
    if key_env is not None:
        return key_env
    return "VLLM_API_KEY" if base_url else "OPENAI_API_KEY"


def _resolved_disable_thinking(model: str, configured: bool | None) -> bool:
    return _is_qwen(model) if configured is None else configured


def main() -> None:
    args = parse_args()
    if not 0.0 < args.lam < 1.0:
        raise SystemExit("--lam must be in (0, 1)")
    if not 0.0 < args.local_validity_threshold <= 1.0:
        raise SystemExit("--local-validity-threshold must be in (0, 1]")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    analyzer_base_url = _resolved_endpoint(
        args.analyzer_model, args.analyzer_base_url
    )
    verifier_base_url = _resolved_endpoint(
        args.verifier_model, args.verifier_base_url
    )
    matcher_base_url = _resolved_endpoint(args.matcher_model, args.matcher_base_url)
    analyzer_key_env = _resolved_key_env(
        analyzer_base_url, args.analyzer_api_key_env
    )
    verifier_key_env = _resolved_key_env(
        verifier_base_url, args.verifier_api_key_env
    )
    matcher_key_env = _resolved_key_env(
        matcher_base_url, args.matcher_api_key_env
    )
    analyzer_disable_thinking = _resolved_disable_thinking(
        args.analyzer_model, args.analyzer_disable_thinking
    )
    verifier_disable_thinking = _resolved_disable_thinking(
        args.verifier_model, args.disable_thinking
    )
    matcher_disable_thinking = _resolved_disable_thinking(
        args.matcher_model, args.matcher_disable_thinking
    )

    examples = {ex.example_id: ex for ex in load_examples(args.data, args.split_name)}

    generations = load_rollout_groups(args.rollouts)
    unknown_ids = sorted(set(generations) - set(examples))
    if unknown_ids:
        preview = ", ".join(unknown_ids[:5])
        raise SystemExit(
            f"{len(unknown_ids)} rollout example IDs are absent from --data "
            f"under split {args.split_name!r} ({preview}); check --split-name"
        )

    out_path = Path(args.out)
    verifier_info = {
        "analyzer_model": args.analyzer_model,
        "verifier_model": args.verifier_model,
        "matcher_model": args.matcher_model,
        "lambda": args.lam,
        "local_validity_threshold": args.local_validity_threshold,
        "analyzer_disable_thinking": analyzer_disable_thinking,
        "verifier_disable_thinking": verifier_disable_thinking,
        "matcher_disable_thinking": matcher_disable_thinking,
        "analyzer_reasoning_effort": args.analyzer_reasoning_effort,
        "verifier_reasoning_effort": args.reasoning_effort,
        "matcher_reasoning_effort": args.matcher_reasoning_effort,
    }

    def is_complete(record: dict) -> bool:
        return "error" not in record and record.get("verifier") == verifier_info and (
            record.get("discarded") is True
            or isinstance(record.get("judgments"), list)
        )

    done = prepare_resume_file(out_path, is_complete)

    todo = [examples[eid] for eid in generations if eid in examples and eid not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(generations)} rollout groups, {len(done)} scored, {len(todo)} to score"
    )

    analyzer_client = OpenAICompatClient(
        model=args.analyzer_model,
        base_url=analyzer_base_url,
        api_key_env=analyzer_key_env,
        disable_thinking=analyzer_disable_thinking,
        reasoning_effort=args.analyzer_reasoning_effort,
    )
    verifier_client = OpenAICompatClient(
        model=args.verifier_model,
        base_url=verifier_base_url,
        api_key_env=verifier_key_env,
        disable_thinking=verifier_disable_thinking,
        reasoning_effort=args.reasoning_effort,
    )
    matcher_client = OpenAICompatClient(
        model=args.matcher_model,
        base_url=matcher_base_url,
        api_key_env=matcher_key_env,
        disable_thinking=matcher_disable_thinking,
        reasoning_effort=args.matcher_reasoning_effort,
    )
    scorer = Scorer(
        client=verifier_client,
        lam=args.lam,
        tau_loc=args.local_validity_threshold,
        analyzer_client=analyzer_client,
        matcher_client=matcher_client,
    )

    def run(example):
        try:
            record = scorer.score_group(example, generations[example.example_id])
            record["verifier"] = verifier_info
            return record
        except Exception as exc:  # noqa: BLE001 - retry on the next run
            return {"example_id": example.example_id, "error": str(exc)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with (
        open(out_path, "a", encoding="utf-8") as f,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for record in pool.map(run, todo):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            wrote += 1
            if wrote % 20 == 0:
                print(f"  {wrote}/{len(todo)}")
    print(f"Done. Wrote {wrote} group records to {out_path}")


if __name__ == "__main__":
    main()
