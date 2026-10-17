"""Score a group of judgments for one example, end to end.

Per example: one analyzer call builds D*; per judgment: one grounding call,
one validity call, one matcher call. All outputs are plain dicts ready for
jsonl serialization.

The pure assembly helpers (`score_judgment`, `unparseable_record`,
`group_record`) are shared with the Batch API path (batch_score.py) so both
paths produce byte-identical score records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMClient
from .parsing import is_parseable, parse_judgment, validation_errors
from .rubrics import criterion_weights, get_rubric
from .scoring import (
    claim_reliabilities,
    coverage,
    criterion_intervals,
    criterion_qualities,
    decision_support,
    maximum_weight_matching,
    q_proc,
    q_total,
    soundness,
)
from .types import JudgmentScores, MaterialIssue, PairExample
from .verify import generate_issues, match_issues, score_grounding, score_validity


def unparseable_record(
    gen: dict[str, Any], lam: float, errors: list[str] | None = None
) -> dict[str, Any]:
    scores = JudgmentScores(
        parse_ok=False,
        outcome=0,
        s_snd=0.0,
        s_cov=0.0,
        s_dec=0.0,
        q_proc=0.0,
        q_total=q_total(0, 0.0, lam),
    )
    record = {"model": gen["model"], "text": gen["text"], **scores.to_dict()}
    if errors:
        record["parse_errors"] = errors
    return record


def discarded_group_record(example: PairExample, reason: str) -> dict[str, Any]:
    """A terminal record for a group excluded before judgment scoring."""
    return {
        "example_id": example.example_id,
        "domain": example.domain,
        "gold": example.gold,
        "discarded": True,
        "discard_reason": reason,
    }


def score_judgment(
    example: PairExample,
    weights: dict[str, float],
    issues: list[MaterialIssue],
    gen: dict[str, Any],
    judgment,
    grounding: dict[int, float],
    validity: dict[int, float],
    pair_scores: list[list[float]],
    lam: float,
    tau_loc: float = 1.0,
) -> dict[str, Any]:
    """Pure score assembly for one parseable judgment."""
    claim_q = claim_reliabilities(judgment, grounding, validity)
    s_snd = soundness(judgment, claim_q, weights)
    matches = maximum_weight_matching(issues, pair_scores, claim_q)
    s_cov = coverage(issues, matches, claim_q)
    intervals = criterion_intervals(judgment, claim_q, weights, tau_loc)
    s_dec = decision_support(judgment, intervals, weights)
    q_k = criterion_qualities(judgment, grounding, validity)
    qp = q_proc(s_snd, s_cov, s_dec)
    outcome = int(judgment.verdict == example.gold)

    scores = JudgmentScores(
        parse_ok=True,
        outcome=outcome,
        s_snd=s_snd,
        s_cov=s_cov,
        s_dec=s_dec,
        q_proc=qp,
        q_total=q_total(outcome, qp, lam),
        detail={
            "verdict": judgment.verdict,
            "grounding": grounding,
            "validity": validity,
            "claim_q": claim_q,
            "match_scores": pair_scores,
            "matches": [list(m) for m in matches],
            "criterion_q": q_k,
            "criterion_intervals": {
                name: list(interval) for name, interval in intervals.items()
            },
            "tau_loc": tau_loc,
            "criteria": [b.name for b in judgment.criteria],
        },
    )
    return {"model": gen["model"], "text": gen["text"], **scores.to_dict()}


def group_record(
    example: PairExample,
    issues: list[MaterialIssue],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "domain": example.domain,
        "gold": example.gold,
        "issues": [
            {
                "description": i.description,
                "criterion": i.criterion,
                "importance": i.importance,
                "favors": i.favors,
            }
            for i in issues
        ],
        "judgments": judgments,
    }


@dataclass
class Scorer:
    # Grounding and criterion-validity verifier.
    client: LLMClient
    lam: float = 0.5
    tau_loc: float = 1.0
    # The analyzer can use a stronger model while the claim-level checks use
    # the main verifier client.
    analyzer_client: LLMClient | None = None
    # Semantic matching can use another model or endpoint.
    matcher_client: LLMClient | None = None

    def score_group(
        self,
        example: PairExample,
        generations: list[dict[str, Any]],
        issues: list[MaterialIssue] | None = None,
    ) -> dict[str, Any]:
        """Score all sampled judgments of one example.

        `generations` items: {"model": str, "text": str}. Returns a group
        record with per-judgment scores and the issue set used. Pass `issues`
        to reuse an already-generated D* instead of calling the analyzer.
        """
        rubric = get_rubric(example.domain)
        weights = criterion_weights(rubric)
        if issues is None:
            issues = generate_issues(
                self.analyzer_client or self.client, example, rubric
            )
        if not issues:
            return discarded_group_record(example, "empty_issue_set")

        judgments = []
        for gen in generations:
            judgment = parse_judgment(gen["text"])
            if not is_parseable(judgment, rubric):
                judgments.append(
                    unparseable_record(
                        gen, self.lam, validation_errors(judgment, rubric)
                    )
                )
                continue

            grounding = score_grounding(self.client, example, judgment)
            validity = score_validity(self.client, example, rubric, judgment)
            pair_scores = match_issues(
                self.matcher_client or self.client, issues, judgment
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
                    self.lam,
                    self.tau_loc,
                )
            )

        return group_record(example, issues, judgments)
