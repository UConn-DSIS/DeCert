"""Core data types shared across the pipeline.

Terminology follows the paper:
- a *PairExample* is one HelpSteer3 preference row presented to the judge;
- a *Judgment* is one sampled judge output (rationale + verdict), parsed into
  observation claims, criterion claims and a decision claim;
- a *MaterialIssue* is one entry of the analyzer-generated issue set D*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Message = dict[str, str]  # {"role": ..., "content": ...}


@dataclass
class PairExample:
    example_id: str
    domain: str
    language: str
    context: list[Message]
    response_a: str
    response_b: str
    gold: str  # "A" or "B": which response the human preference favors
    swapped: bool  # True if response_a is the source row's response2
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Observation claim: what a response contains / lacks, with evidence."""

    side: str  # "A" or "B"
    text: str


@dataclass
class CriterionBlock:
    """One criterion's observations plus its criterion claim (winner + why)."""

    name: str
    observations: list[Observation]
    winner: str  # "A", "B" or "tie"
    reasoning: str


@dataclass
class Judgment:
    raw: str
    criteria: list[CriterionBlock]
    decision_text: str
    verdict: str | None  # "A", "B" or None when unparseable


@dataclass
class MaterialIssue:
    description: str
    criterion: str
    importance: float  # rho_t
    favors: str  # "A" or "B"


@dataclass
class JudgmentScores:
    parse_ok: bool
    outcome: int  # O in {0, 1}; 0 also covers unparseable verdicts
    s_snd: float
    s_cov: float
    s_dec: float
    q_proc: float
    q_total: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
