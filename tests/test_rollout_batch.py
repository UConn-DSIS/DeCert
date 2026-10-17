import json
import sys

import pytest

from decert.data import read_jsonl
from rollout.merge_openai_batch import main


def batch_record(custom_id, text=None, error=None):
    response = None
    if text is not None:
        response = {"body": {"choices": [{"message": {"content": text}}]}}
    return {"custom_id": custom_id, "response": response, "error": error}


def run_merge(monkeypatch, input_path, output_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_openai_batch.py",
            "--batch-output",
            str(input_path),
            "--expected-n",
            "2",
            "--out",
            str(output_path),
            *extra,
        ],
    )
    main()


def test_merge_batch_requires_every_repetition(tmp_path, monkeypatch):
    input_path = tmp_path / "batch_output.jsonl"
    records = [
        batch_record("train:0__rep0", "first"),
        batch_record("train:0__rep1", error={"message": "temporary"}),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(SystemExit, match="do not have 2 successful repetitions"):
        run_merge(monkeypatch, input_path, tmp_path / "rollout.jsonl")


def test_merge_batch_complete(tmp_path, monkeypatch):
    input_path = tmp_path / "batch_output.jsonl"
    records = [
        batch_record("train:0__rep0", "first"),
        batch_record("train:0__rep1", "second"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    output_path = tmp_path / "rollout.jsonl"

    run_merge(
        monkeypatch,
        input_path,
        output_path,
        "--expected-examples",
        "1",
    )

    assert read_jsonl(output_path)[0]["generation"] == ["first", "second"]
