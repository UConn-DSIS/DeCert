"""Batch API version of the scoring stage.

The verifier calls have one dependency: the matcher needs the analyzer's
issue set. So scoring splits into two batch phases plus a local assembly:

  phase 1 (independent): analyzer (1/example) + grounding + validity
                         (1 each per parseable judgment)
  phase 2 (needs phase-1 analyzer output): matcher (1 per parseable
                         judgment; the group is discarded when D* is empty)
  assemble (local, no API): parse all verifier outputs and compute scores

custom_id scheme, all keyed to the merged-group judgment index j
(order fixed by load_rollout_groups):
  <example_id>|analyzer   <example_id>|g<j>   <example_id>|v<j>   <example_id>|m<j>

The CLI checks that every requested result is present before it assembles
scores. A present but malformed grounding, validity, or matcher response
parses to a pessimistic score (0.0 / no match). A present but malformed or
empty analyzer response discards the group, matching the synchronous path.
Assembly reuses pipeline.score_judgment, so batch and sync scores are
computed by the same code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .parsing import is_parseable, parse_judgment, validation_errors
from .pipeline import (
    discarded_group_record,
    group_record,
    score_judgment,
    unparseable_record,
)
from .prompts import (
    build_analyzer_messages,
    build_grounding_messages,
    build_matcher_messages,
    build_validity_messages,
    numbered_observations,
)
from .rubrics import criterion_weights, get_rubric
from .types import Message, PairExample
from .verify import (
    VERIFIER_PARAMS,
    parse_grounding_response,
    parse_issues_response,
    parse_match_response,
    parse_validity_response,
)

Groups = dict[str, list[dict[str, str]]]


def _request(custom_id: str, messages: list[Message], model: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, **VERIFIER_PARAMS},
    }


def read_batch_texts(paths: Iterable[str | Path]) -> dict[str, str]:
    """Batch result files -> {custom_id: response text}; errored -> absent."""
    texts: dict[str, str] = {}
    for path in paths:
        for rec in read_jsonl(path):
            if rec.get("error") or rec.get("response") is None:
                continue
            choices = (rec["response"].get("body") or {}).get("choices") or []
            if choices:
                custom_id = rec["custom_id"]
                if custom_id in texts:
                    raise ValueError(f"duplicate batch result for {custom_id!r}")
                texts[custom_id] = choices[0]["message"]["content"] or ""
    return texts


def required_phase1_ids(examples: dict[str, PairExample], groups: Groups) -> set[str]:
    """Result IDs that must be present before phase 2 or assembly."""
    required = set()
    for example_id, gens in groups.items():
        rubric = get_rubric(examples[example_id].domain)
        required.add(f"{example_id}|analyzer")
        for j, gen in enumerate(gens):
            judgment = parse_judgment(gen["text"])
            if not is_parseable(judgment, rubric):
                continue
            if numbered_observations(judgment):
                required.add(f"{example_id}|g{j}")
            required.add(f"{example_id}|v{j}")
    return required


def required_phase2_ids(
    examples: dict[str, PairExample],
    groups: Groups,
    phase1_texts: dict[str, str],
) -> set[str]:
    """Matcher result IDs required for analyzers with non-empty issue sets."""
    required = set()
    for example_id, gens in groups.items():
        rubric = get_rubric(examples[example_id].domain)
        issues = parse_issues_response(
            phase1_texts.get(f"{example_id}|analyzer", ""), rubric
        )
        if not issues:
            continue
        for j, gen in enumerate(gens):
            judgment = parse_judgment(gen["text"])
            if is_parseable(judgment, rubric):
                required.add(f"{example_id}|m{j}")
    return required


def phase1_requests(
    examples: dict[str, PairExample],
    groups: Groups,
    model: str,
    analyzer_model: str | None = None,
) -> list[dict[str, Any]]:
    analyzer_model = analyzer_model or model
    requests = []
    for example_id, gens in groups.items():
        example = examples[example_id]
        rubric = get_rubric(example.domain)
        requests.append(
            _request(
                f"{example_id}|analyzer",
                build_analyzer_messages(example, rubric),
                analyzer_model,
            )
        )
        for j, gen in enumerate(gens):
            judgment = parse_judgment(gen["text"])
            if not is_parseable(judgment, rubric):
                continue
            if numbered_observations(judgment):
                requests.append(
                    _request(
                        f"{example_id}|g{j}",
                        build_grounding_messages(example, judgment),
                        model,
                    )
                )
            requests.append(
                _request(
                    f"{example_id}|v{j}",
                    build_validity_messages(example, rubric, judgment),
                    model,
                )
            )
    return requests


def phase2_requests(
    examples: dict[str, PairExample],
    groups: Groups,
    phase1_texts: dict[str, str],
    model: str,
) -> list[dict[str, Any]]:
    requests = []
    for example_id, gens in groups.items():
        rubric = get_rubric(examples[example_id].domain)
        issues = parse_issues_response(
            phase1_texts.get(f"{example_id}|analyzer", ""), rubric
        )
        if not issues:
            continue
        for j, gen in enumerate(gens):
            judgment = parse_judgment(gen["text"])
            if not is_parseable(judgment, rubric):
                continue
            requests.append(
                _request(
                    f"{example_id}|m{j}",
                    build_matcher_messages(issues, judgment),
                    model,
                )
            )
    return requests


def assemble_groups(
    examples: dict[str, PairExample],
    groups: Groups,
    phase1_texts: dict[str, str],
    phase2_texts: dict[str, str],
    lam: float = 0.5,
    tau_loc: float = 1.0,
) -> list[dict[str, Any]]:
    records = []
    for example_id, gens in groups.items():
        example = examples[example_id]
        rubric = get_rubric(example.domain)
        weights = criterion_weights(rubric)
        issues = parse_issues_response(
            phase1_texts.get(f"{example_id}|analyzer", ""), rubric
        )
        if not issues:
            records.append(discarded_group_record(example, "empty_issue_set"))
            continue

        judgments = []
        for j, gen in enumerate(gens):
            judgment = parse_judgment(gen["text"])
            if not is_parseable(judgment, rubric):
                judgments.append(
                    unparseable_record(gen, lam, validation_errors(judgment, rubric))
                )
                continue
            grounding = parse_grounding_response(
                phase1_texts.get(f"{example_id}|g{j}", ""), judgment
            )
            validity = parse_validity_response(
                phase1_texts.get(f"{example_id}|v{j}", ""), judgment
            )
            pair_scores = parse_match_response(
                phase2_texts.get(f"{example_id}|m{j}", ""), issues, judgment
            )
            judgments.append(
                score_judgment(
                    example,
                    weights,
                    issues,
                    gen,
                    judgment,
                    grounding,
                    validity,
                    pair_scores,
                    lam,
                    tau_loc,
                )
            )
        records.append(group_record(example, issues, judgments))
    return records


# OpenAI Batch API limits: 50k requests and 200MB per file.
MAX_REQUESTS_PER_SHARD = 50_000
MAX_BYTES_PER_SHARD = 150 * 1024 * 1024


def write_sharded_requests(
    out_dir: str | Path, prefix: str, requests: list[dict[str, Any]]
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    shard_lines: list[str] = []
    shard_bytes = 0

    def flush() -> None:
        nonlocal shard_lines, shard_bytes
        if not shard_lines:
            return
        path = out_dir / f"{prefix}_{len(paths):03d}.jsonl"
        path.write_text("".join(shard_lines), encoding="utf-8")
        paths.append(path)
        shard_lines, shard_bytes = [], 0

    for req in requests:
        line = json.dumps(req, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))
        if shard_lines and (
            len(shard_lines) >= MAX_REQUESTS_PER_SHARD
            or shard_bytes + size > MAX_BYTES_PER_SHARD
        ):
            flush()
        shard_lines.append(line)
        shard_bytes += size
    flush()
    return paths
