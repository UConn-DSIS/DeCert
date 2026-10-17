"""Training-data construction from scored judgment groups.

A group is the pool of sampled judgments for one input (e.g. 3 GPT-4o +
5 Qwen3-8B rollouts). Only parseable judgments are used.

- SFT targets: per group, the verdict-correct judgment with the highest
  Q_proc; groups without a correct judgment are skipped.
- Max-Min DPO pairs: chosen = argmax Q_total, rejected = argmin Q_total of
  the same group. Q_total is outcome-dominant, so
    * mixed groups (correct and wrong verdicts) yield outcome pairs
      (correct vs. wrong), kept unconditionally;
    * all-correct groups yield same-verdict rationale preference pairs, kept
      only when the Q_proc gap exceeds a threshold delta;
    * all-wrong groups are discarded.
"""

from __future__ import annotations

from typing import Any

from .prompts import build_judge_messages
from .rubrics import get_rubric
from .scoring import q_total
from .types import PairExample

PROCESS_AGGREGATIONS = ("geometric", "arithmetic")


def process_score(judgment: dict[str, Any], aggregation: str) -> float:
    """Process score used to rank preference candidates."""
    if aggregation == "geometric":
        return judgment["q_proc"]
    if aggregation == "arithmetic":
        return sum(judgment[name] for name in ("s_snd", "s_cov", "s_dec")) / 3.0
    raise ValueError(f"unknown process aggregation {aggregation!r}")


def total_score(judgment: dict[str, Any], aggregation: str) -> float:
    """Outcome-dominant score under the selected process aggregation."""
    if aggregation == "geometric":
        return judgment["q_total"]
    return q_total(judgment["outcome"], process_score(judgment, aggregation))


def usable_judgments(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [j for j in group.get("judgments", []) if j.get("parse_ok")]


def _pair_record(
    example: PairExample,
    group: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    process_aggregation: str = "geometric",
) -> dict[str, Any]:
    return {
        "example_id": group["example_id"],
        "domain": group["domain"],
        "gold": group["gold"],
        "pair_type": pair_type,
        "process_aggregation": process_aggregation,
        "prompt": build_judge_messages(example, get_rubric(example.domain)),
        "chosen": chosen["text"],
        "rejected": rejected["text"],
        "chosen_model": chosen["model"],
        "rejected_model": rejected["model"],
        "chosen_scores": _score_fields(chosen, process_aggregation),
        "rejected_scores": _score_fields(rejected, process_aggregation),
    }


def build_sft_targets(
    groups: list[dict[str, Any]],
    examples_by_id: dict[str, PairExample],
    min_qproc: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """SFT targets: per group, the argmax-Q_proc verdict-correct judgment.

    SFT needs no outcome contrast, so all-correct groups qualify. An optional
    min_qproc floor drops groups whose best certificate is below it (0.0 =
    verdict correctness is the only requirement, as in the paper)."""
    records = []
    stats = {"kept": 0, "no_correct": 0, "below_min_qproc": 0}

    for group in groups:
        correct = [j for j in usable_judgments(group) if j["outcome"] == 1]
        if not correct:
            stats["no_correct"] += 1
            continue
        target = max(correct, key=lambda j: j["q_proc"])
        if target["q_proc"] < min_qproc:
            stats["below_min_qproc"] += 1
            continue
        example = examples_by_id[group["example_id"]]
        records.append(
            {
                "example_id": group["example_id"],
                "domain": group["domain"],
                "gold": group["gold"],
                "prompt": build_judge_messages(example, get_rubric(example.domain)),
                "target": target["text"],
                "target_model": target["model"],
                "target_scores": _score_fields(target),
            }
        )
        stats["kept"] += 1

    return records, stats


def build_maxmin_pairs(
    groups: list[dict[str, Any]],
    examples_by_id: dict[str, PairExample],
    same_verdict_delta: float = 0.25,
    process_aggregation: str = "geometric",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Max-Min pairs under the selected process-score aggregation.

    Chosen = argmax total score and rejected = argmin total score of the
    same group. The total score is outcome-dominant, so mixed groups reduce to
    the outcome pair and are kept unconditionally — the verdict-correctness
    margin is the signal. All-correct groups yield same-verdict certificate
    pairs, kept only when their process-score gap exceeds same_verdict_delta:
    a near-tied certificate pair is verifier noise, not preference.
    All-wrong groups are skipped (chosen must be verdict-correct)."""
    records = []
    stats = {
        "kept": 0,
        "fewer_than_two_parseable": 0,
        "all_wrong": 0,
        "same_verdict_below_delta": 0,
    }

    for group in groups:
        judgments = usable_judgments(group)
        if len(judgments) < 2:
            stats["fewer_than_two_parseable"] += 1
            continue
        chosen = max(judgments, key=lambda j: total_score(j, process_aggregation))
        rejected = min(judgments, key=lambda j: total_score(j, process_aggregation))
        if chosen["outcome"] == 0:  # max Q_total wrong => the whole group is
            stats["all_wrong"] += 1
            continue
        if (
            rejected["outcome"] == 1  # all-correct group: same-verdict pair
            and process_score(chosen, process_aggregation)
            - process_score(rejected, process_aggregation)
            <= same_verdict_delta
        ):
            stats["same_verdict_below_delta"] += 1
            continue

        example = examples_by_id[group["example_id"]]
        records.append(
            _pair_record(
                example,
                group,
                chosen,
                rejected,
                "maxmin",
                process_aggregation,
            )
        )
        stats["kept"] += 1

    return records, stats


def _score_fields(
    judgment: dict[str, Any], process_aggregation: str = "geometric"
) -> dict[str, float]:
    fields = {
        k: judgment[k]
        for k in ("outcome", "s_snd", "s_cov", "s_dec", "q_proc", "q_total")
    }
    fields["process_score"] = process_score(judgment, process_aggregation)
    fields["total_score"] = total_score(judgment, process_aggregation)
    return fields
