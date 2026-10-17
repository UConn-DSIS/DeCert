"""Evaluation policy (rubrics) for HelpSteer3 domains.

Each domain gets an ordered list of criteria with policy importance alpha_k
(weights sum to 1 per domain). These weights are a design choice of our
evaluation policy — they are used identically in the judge prompt, the
material-issue analyzer, and the decision-support aggregation, so the whole
pipeline is conditioned on one policy.

The policy is correctness-dominant: a response with a material factual or
code error must not win on style alone. That mirrors HelpSteer3's annotation
guidance where helpfulness is grounded in being correct and on-task.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    name: str
    weight: float  # alpha_k, policy importance
    description: str


CORRECTNESS = Criterion(
    "correctness",
    0.0,  # weight set per domain below
    "Factual and technical accuracy: claims, facts, calculations, and (for "
    "code) program logic and API usage are correct; nothing is fabricated or "
    "hallucinated.",
)
INSTRUCTION_FOLLOWING = Criterion(
    "instruction_following",
    0.0,
    "The response addresses the user's actual request in the full "
    "conversation, respecting explicit constraints (format, scope, length, "
    "requested language) and the intent of the final user turn.",
)
COMPLETENESS = Criterion(
    "completeness",
    0.0,
    "The response covers the information or steps the user needs for their "
    "goal; no decisive omissions such as missing cases, missing caveats, or "
    "an unanswered part of the question.",
)
CLARITY = Criterion(
    "clarity",
    0.0,
    "Organization, readability, and an appropriate level of detail; no "
    "confusing structure, redundancy, or excessive verbosity.",
)
LANGUAGE_CONSISTENCY = Criterion(
    "language_consistency",
    0.0,
    "The response uses the language the user expects (normally the language "
    "of the user's last message) with natural, fluent usage.",
)


def _weighted(criterion: Criterion, weight: float) -> Criterion:
    return Criterion(criterion.name, weight, criterion.description)


RUBRICS: dict[str, list[Criterion]] = {
    "code": [
        _weighted(CORRECTNESS, 0.40),
        _weighted(INSTRUCTION_FOLLOWING, 0.25),
        _weighted(COMPLETENESS, 0.20),
        _weighted(CLARITY, 0.15),
    ],
    "stem": [
        _weighted(CORRECTNESS, 0.40),
        _weighted(INSTRUCTION_FOLLOWING, 0.25),
        _weighted(COMPLETENESS, 0.20),
        _weighted(CLARITY, 0.15),
    ],
    "general": [
        _weighted(CORRECTNESS, 0.30),
        _weighted(INSTRUCTION_FOLLOWING, 0.30),
        _weighted(COMPLETENESS, 0.25),
        _weighted(CLARITY, 0.15),
    ],
    "multilingual": [
        _weighted(CORRECTNESS, 0.30),
        _weighted(INSTRUCTION_FOLLOWING, 0.25),
        _weighted(LANGUAGE_CONSISTENCY, 0.20),
        _weighted(COMPLETENESS, 0.15),
        _weighted(CLARITY, 0.10),
    ],
}

def get_rubric(domain: str) -> list[Criterion]:
    if domain not in RUBRICS:
        raise KeyError(f"no rubric for domain {domain!r}")
    return RUBRICS[domain]


def criterion_weights(rubric: list[Criterion]) -> dict[str, float]:
    return {c.name: c.weight for c in rubric}


def rubric_text(rubric: list[Criterion]) -> str:
    """Human-readable policy block shared by judge / analyzer prompts."""
    lines = []
    for c in rubric:
        lines.append(f"- {c.name} (importance {c.weight:.2f}): {c.description}")
    lines.append(
        "Aggregation policy: weigh criteria by their importance. A material "
        "correctness failure outweighs advantages in style or presentation."
    )
    return "\n".join(lines)
