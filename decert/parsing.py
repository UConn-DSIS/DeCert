"""Parse judge outputs (the tagged certificate format) into Judgment objects.

Parsing never raises: malformed pieces are dropped and a missing/invalid
<answer> tag yields verdict None, which downstream scoring treats as an
incorrect outcome.
"""

from __future__ import annotations

import re
from collections import Counter

from .rubrics import Criterion
from .types import CriterionBlock, Judgment, Observation

# Reasoning-model think blocks (e.g. Qwen3) are stripped before parsing so
# draft tags inside them can never be mistaken for the certificate. An
# unclosed block (truncated generation) swallows the rest of the text, which
# correctly yields an unparseable judgment.
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)

COMPARISON_RE = re.compile(r"<comparison>(.*?)</comparison>", re.IGNORECASE | re.DOTALL)

CRITERION_RE = re.compile(
    r'<criterion\s+name="([^"]+)"\s*>(.*?)</criterion>', re.IGNORECASE | re.DOTALL
)
OBS_RE = re.compile(r'<obs\s+side="([AB])"\s*>(.*?)</obs>', re.IGNORECASE | re.DOTALL)
RESULT_RE = re.compile(
    r'<result\s+winner="(A|B|tie)"\s*>(.*?)</result>', re.IGNORECASE | re.DOTALL
)
DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(
    r"<answer>\s*\[\[\s*([AB])\s*\]\]\s*</answer>", re.IGNORECASE | re.DOTALL
)
CRITERION_OPEN_RE = re.compile(r"<criterion\b", re.IGNORECASE)
CRITERION_CLOSE_RE = re.compile(r"</criterion\s*>", re.IGNORECASE)
OBS_OPEN_RE = re.compile(r"<obs\b", re.IGNORECASE)
OBS_CLOSE_RE = re.compile(r"</obs\s*>", re.IGNORECASE)
RESULT_OPEN_RE = re.compile(r"<result\b", re.IGNORECASE)
RESULT_CLOSE_RE = re.compile(r"</result\s*>", re.IGNORECASE)
COMPARISON_OPEN_RE = re.compile(r"<comparison\b", re.IGNORECASE)
COMPARISON_CLOSE_RE = re.compile(r"</comparison\s*>", re.IGNORECASE)
DECISION_OPEN_RE = re.compile(r"<decision\b", re.IGNORECASE)
DECISION_CLOSE_RE = re.compile(r"</decision\s*>", re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer\b", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer\s*>", re.IGNORECASE)


def parse_judgment(text: str) -> Judgment:
    raw = text or ""
    text = THINK_RE.sub("", raw)
    comparison = COMPARISON_RE.search(text)
    comparison_body = comparison.group(1) if comparison else ""
    criteria = []
    for name, body in CRITERION_RE.findall(comparison_body):
        observations = [
            Observation(side=side.upper(), text=obs_text.strip())
            for side, obs_text in OBS_RE.findall(body)
        ]
        result = RESULT_RE.search(body)
        if result is None:
            # A criterion without a conclusion is not a usable claim block.
            continue
        winner = result.group(1)
        winner = "tie" if winner.lower() == "tie" else winner.upper()
        criteria.append(
            CriterionBlock(
                name=name.strip().lower(),
                observations=observations,
                winner=winner,
                reasoning=result.group(2).strip(),
            )
        )

    decision = DECISION_RE.search(text)
    answer = ANSWER_RE.search(text)
    return Judgment(
        raw=raw,
        criteria=criteria,
        decision_text=decision.group(1).strip() if decision else "",
        verdict=answer.group(1).upper() if answer else None,
    )


def validation_errors(
    judgment: Judgment, rubric: list[Criterion] | None = None
) -> list[str]:
    """Return structural and policy errors in a parsed certificate.

    When a rubric is supplied, every policy criterion must occur exactly once.
    This makes omitted, duplicated, and invented criteria explicit instead of
    letting the judge change the policy weights through its output structure.
    """
    errors: list[str] = []
    text = THINK_RE.sub("", judgment.raw or "")

    if (
        len(COMPARISON_RE.findall(text)) != 1
        or len(COMPARISON_OPEN_RE.findall(text)) != 1
        or len(COMPARISON_CLOSE_RE.findall(text)) != 1
    ):
        errors.append("expected exactly one <comparison> block")
    if (
        len(DECISION_RE.findall(text)) != 1
        or len(DECISION_OPEN_RE.findall(text)) != 1
        or len(DECISION_CLOSE_RE.findall(text)) != 1
        or not judgment.decision_text
    ):
        errors.append("expected exactly one non-empty <decision> block")
    if (
        len(ANSWER_RE.findall(text)) != 1
        or len(ANSWER_OPEN_RE.findall(text)) != 1
        or len(ANSWER_CLOSE_RE.findall(text)) != 1
        or judgment.verdict not in ("A", "B")
    ):
        errors.append("expected exactly one <answer> containing [[A]] or [[B]]")
    if not judgment.criteria:
        errors.append("expected at least one complete <criterion> block")

    comparison_match = COMPARISON_RE.search(text)
    if comparison_match is not None:
        comparison_body = comparison_match.group(1)
        raw_blocks = CRITERION_RE.findall(comparison_body)
        if (
            len(CRITERION_OPEN_RE.findall(comparison_body)) != len(raw_blocks)
            or len(CRITERION_CLOSE_RE.findall(comparison_body)) != len(raw_blocks)
        ):
            errors.append("malformed or unclosed <criterion> block")

        for name, body in raw_blocks:
            results = RESULT_RE.findall(body)
            if (
                len(results) != 1
                or len(RESULT_OPEN_RE.findall(body)) != 1
                or len(RESULT_CLOSE_RE.findall(body)) != 1
            ):
                errors.append(
                    f"criterion {name.strip().lower()!r} needs exactly one "
                    "well-formed <result>"
                )

            observations = OBS_RE.findall(body)
            if (
                len(OBS_OPEN_RE.findall(body)) != len(observations)
                or len(OBS_CLOSE_RE.findall(body)) != len(observations)
            ):
                errors.append(
                    f"criterion {name.strip().lower()!r} has a malformed "
                    "<obs> block"
                )

    names = [block.name for block in judgment.criteria]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        errors.append("duplicate criteria: " + ", ".join(duplicates))

    for block in judgment.criteria:
        sides = {obs.side for obs in block.observations if obs.text}
        if "A" not in sides or "B" not in sides:
            errors.append(
                f"criterion {block.name!r} needs a non-empty observation "
                "for both A and B"
            )
        if not block.reasoning:
            errors.append(f"criterion {block.name!r} has an empty result")

    if rubric is not None:
        required = {criterion.name for criterion in rubric}
        present = set(names)
        unknown = sorted(present - required)
        missing = sorted(required - present)
        if unknown:
            errors.append("criteria outside the policy: " + ", ".join(unknown))
        if missing:
            errors.append("missing policy criteria: " + ", ".join(missing))

    return errors


def is_parseable(judgment: Judgment, rubric: list[Criterion] | None = None) -> bool:
    """Whether a judgment is a complete, structurally valid certificate."""
    return not validation_errors(judgment, rubric)
