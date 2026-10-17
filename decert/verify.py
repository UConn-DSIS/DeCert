"""LLM-backed verifiers: grounding, local validity, analyzer (D*), matcher.

All verifier calls request strict JSON; `extract_json_array` tolerates
markdown fences and surrounding prose. Missing or malformed grounding,
validity, and matcher entries default to the most pessimistic score (0.0 / no
match). An empty or malformed analyzer result discards the whole example
because coverage is undefined without a material-issue set.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from .prompts import (
    build_analyzer_messages,
    build_grounding_messages,
    build_matcher_messages,
    build_validity_messages,
    numbered_claims,
    numbered_observations,
)
from .rubrics import Criterion
from .scoring import GROUNDING_VALUE, MATCH_VALUE, VALIDITY_VALUE
from .types import Judgment, MaterialIssue, PairExample

VERIFIER_PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096}


def extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of a model response; [] on failure."""
    decoder = json.JSONDecoder()
    text = text or ""
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def parse_grounding_response(text: str, judgment: Judgment) -> dict[int, float]:
    valid_idx = {idx for idx, *_ in numbered_observations(judgment)}
    scores: dict[int, float] = {}
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        idx = item.get("claim")
        if idx in valid_idx:
            scores[idx] = GROUNDING_VALUE.get(str(item.get("supported")).lower(), 0.0)
    return scores


def score_grounding(
    client: LLMClient, example: PairExample, judgment: Judgment
) -> dict[int, float]:
    """g_j for each observation claim, keyed by flat 1-based index."""
    if not numbered_observations(judgment):
        return {}
    text = client.complete(
        build_grounding_messages(example, judgment), **VERIFIER_PARAMS
    )
    return parse_grounding_response(text, judgment)


def parse_validity_response(text: str, judgment: Judgment) -> dict[int, float]:
    n_blocks = len(judgment.criteria)
    scores: dict[int, float] = {}
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        idx = item.get("block")
        if isinstance(idx, int) and 1 <= idx <= n_blocks:
            scores[idx] = VALIDITY_VALUE.get(str(item.get("follows")).lower(), 0.0)
    return scores


def score_validity(
    client: LLMClient,
    example: PairExample,
    rubric: list[Criterion],
    judgment: Judgment,
) -> dict[int, float]:
    """l_k for each criterion block, keyed by 1-based block index."""
    if not judgment.criteria:
        return {}
    text = client.complete(
        build_validity_messages(example, rubric, judgment), **VERIFIER_PARAMS
    )
    return parse_validity_response(text, judgment)


def parse_issues_response(
    text: str, rubric: list[Criterion] | None = None
) -> list[MaterialIssue]:
    allowed_criteria = {criterion.name for criterion in rubric} if rubric else None
    issues = []
    for item in extract_json_array(text):
        if not isinstance(item, dict) or not item.get("issue"):
            continue
        criterion = str(item.get("criterion", "")).lower()
        favors = str(item.get("favors", "")).upper()
        if not criterion or (
            allowed_criteria is not None and criterion not in allowed_criteria
        ):
            continue
        if favors not in ("A", "B"):
            continue
        try:
            importance = float(item.get("importance", 1))
        except (TypeError, ValueError):
            importance = 1.0
        issues.append(
            MaterialIssue(
                description=str(item["issue"]),
                criterion=criterion,
                importance=min(max(importance, 1.0), 3.0),
                favors=favors,
            )
        )
    return issues[:6]


def generate_issues(
    client: LLMClient, example: PairExample, rubric: list[Criterion]
) -> list[MaterialIssue]:
    """Build the material issue set D*. The analyzer never sees a judgment."""
    text = client.complete(build_analyzer_messages(example, rubric), **VERIFIER_PARAMS)
    return parse_issues_response(text, rubric)


def parse_match_response(
    text: str, issues: list[MaterialIssue], judgment: Judgment
) -> list[list[float]]:
    """Parse the dense issue--claim semantic match matrix.

    Every malformed or missing cell receives zero. The legacy single-match
    format is accepted as a sparse row so partially completed runs fail
    pessimistically rather than crashing.
    """
    n_claims = len(numbered_claims(judgment))
    matrix = [[0.0] * n_claims for _ in issues]
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        issue_idx = item.get("issue")
        if not (isinstance(issue_idx, int) and 1 <= issue_idx <= len(issues)):
            continue
        scores = item.get("scores")
        if isinstance(scores, list):
            for claim_zero_idx, label in enumerate(scores[:n_claims]):
                matrix[issue_idx - 1][claim_zero_idx] = MATCH_VALUE.get(
                    str(label).lower(), 0.0
                )
            continue

        # Backward-compatible parsing for the former top-1 response format.
        claim_idx = item.get("claim")
        if isinstance(claim_idx, int) and 1 <= claim_idx <= n_claims:
            matrix[issue_idx - 1][claim_idx - 1] = MATCH_VALUE.get(
                str(item.get("match")).lower(), 0.0
            )
    return matrix


def match_issues(
    client: LLMClient, issues: list[MaterialIssue], judgment: Judgment
) -> list[list[float]]:
    """Score all issue--claim pairs via the matcher LLM."""
    if not issues or not judgment.criteria:
        return [[0.0] * len(numbered_claims(judgment)) for _ in issues]
    text = client.complete(build_matcher_messages(issues, judgment), **VERIFIER_PARAMS)
    return parse_match_response(text, issues, judgment)
