"""Load exported HelpSteer3 preference rows into PairExample objects.

Each local JSONL row keeps the original fields: domain, language, context,
response1, response2, overall_preference, and individual_preference.

Sign semantics (HelpSteer3 data card): negative overall_preference favors
response1, positive favors response2; 0 was already excluded upstream.

To avoid teaching the judge a position bias, half of the examples present
response2 as "Response A" (deterministic per example_id, so every stage of the
pipeline reconstructs the same ordering).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .types import PairExample


def read_jsonl(
    path: str | Path, *, allow_truncated_last_line: bool = False
) -> list[dict[str, Any]]:
    records = []
    pending_error: tuple[int, json.JSONDecodeError] | None = None
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if pending_error is not None:
                bad_line, exc = pending_error
                raise ValueError(
                    f"invalid JSON in {path} at line {bad_line}: {exc.msg}"
                ) from exc
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if allow_truncated_last_line:
                    pending_error = (line_number, exc)
                    continue
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"JSONL record in {path} at line {line_number} is not an object"
                )
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def prepare_resume_file(
    path: str | Path, is_complete: Callable[[dict[str, Any]], bool]
) -> set[str]:
    """Keep one complete record per example and return completed IDs.

    Long API jobs append one record at a time. If a prior run wrote an error
    record or was interrupted after writing duplicate records, those entries
    must not make the next run skip the example or duplicate its generations.
    The last complete record for an example is retained.
    """
    path = Path(path)
    if not path.exists():
        return set()

    truncated_tail = False
    try:
        records = read_jsonl(path)
    except ValueError:
        records = read_jsonl(path, allow_truncated_last_line=True)
        truncated_tail = True
    retained: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not is_complete(record):
            continue
        if example_id not in retained:
            order.append(example_id)
        retained[example_id] = record

    cleaned = [retained[example_id] for example_id in order]
    if truncated_tail or cleaned != records:
        temp_path = path.with_name(path.name + ".tmp")
        write_jsonl(temp_path, cleaned)
        temp_path.replace(path)
    return set(retained)


def gold_side(overall_preference: int) -> str:
    """Which source response the human preference favors ("1" or "2")."""
    if overall_preference == 0:
        raise ValueError("overall_preference == 0 has no winner; filter upstream")
    return "1" if overall_preference < 0 else "2"


def should_swap(example_id: str) -> bool:
    """Deterministic 50% A/B swap keyed on example_id."""
    digest = hashlib.sha1(example_id.encode("utf-8")).digest()
    return digest[0] % 2 == 1


def make_example(record: dict[str, Any], example_id: str) -> PairExample:
    winner = gold_side(int(record["overall_preference"]))
    swapped = should_swap(example_id)
    if swapped:
        response_a, response_b = record["response2"], record["response1"]
        gold = "A" if winner == "2" else "B"
    else:
        response_a, response_b = record["response1"], record["response2"]
        gold = "A" if winner == "1" else "B"
    return PairExample(
        example_id=example_id,
        domain=record["domain"],
        language=record["language"],
        context=list(record["context"]),
        response_a=response_a,
        response_b=response_b,
        gold=gold,
        swapped=swapped,
        meta={"overall_preference": record["overall_preference"]},
    )


def load_rollout_groups(paths: Iterable[str | Path]) -> dict[str, list[dict[str, str]]]:
    """Merge rollout files into {example_id: [{"model", "text"}, ...]}.

    Generation order is determined by the order of `paths` and the order
    within each file — every stage (sync scoring, batch phases, assembly)
    must pass the same rollout files in the same order so judgment indices
    line up.
    """
    groups: dict[str, list[dict[str, str]]] = {}
    loaded_paths: set[Path] = set()
    for path in paths:
        resolved_path = Path(path).resolve()
        if resolved_path in loaded_paths:
            raise ValueError(f"rollout file passed more than once: {path}")
        loaded_paths.add(resolved_path)

        seen_in_file: set[str] = set()
        for rec in read_jsonl(path):
            example_id = rec.get("example_id")
            if not isinstance(example_id, str):
                raise TypeError(f"rollout record in {path} has no string example_id")
            if example_id in seen_in_file:
                raise ValueError(
                    f"duplicate example_id {example_id!r} in rollout file {path}"
                )
            seen_in_file.add(example_id)

            model = rec.get("model")
            generations = rec.get("generation")
            if not isinstance(model, str) or not isinstance(generations, list):
                raise TypeError(
                    f"invalid rollout record for {example_id!r} in {path}"
                )
            if not all(isinstance(text, str) for text in generations):
                raise TypeError(
                    f"non-string generation for {example_id!r} in {path}"
                )

            bucket = groups.setdefault(example_id, [])
            for text in generations:
                bucket.append({"model": model, "text": text})
    return groups


def load_examples(path: str | Path, split_name: str) -> list[PairExample]:
    """Load a HelpSteer3 JSONL file; id is ``<split_name>:<line_no>``."""
    examples = []
    for idx, record in enumerate(read_jsonl(path)):
        examples.append(make_example(record, f"{split_name}:{idx}"))
    return examples


def load_scored_groups(
    paths: Iterable[str | Path], examples_by_id: dict[str, PairExample]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load usable scored groups and reject stale or duplicated records."""
    groups: list[dict[str, Any]] = []
    stats = {"errors": 0, "discarded": 0}
    terminal_ids: set[str] = set()

    for path in paths:
        for record in read_jsonl(path):
            if "error" in record:
                stats["errors"] += 1
                continue

            example_id = record.get("example_id")
            if not isinstance(example_id, str):
                raise TypeError(f"scored record in {path} has no string example_id")
            if example_id in terminal_ids:
                raise ValueError(
                    f"duplicate terminal scored record for {example_id!r} in {path}"
                )
            terminal_ids.add(example_id)

            if example_id not in examples_by_id:
                raise ValueError(
                    f"scored example_id {example_id!r} is absent from the input "
                    "data; check --data and --split-name"
                )

            example = examples_by_id[example_id]
            if record.get("domain") != example.domain or record.get("gold") != example.gold:
                raise ValueError(
                    f"scored metadata for {example_id!r} does not match the input data"
                )
            if record.get("discarded"):
                stats["discarded"] += 1
                continue
            if not isinstance(record.get("judgments"), list):
                raise TypeError(
                    f"scored record for {example_id!r} has no judgments list"
                )
            groups.append(record)

    return groups, stats
