import json

import pytest

from decert.data import (
    gold_side,
    load_examples,
    load_rollout_groups,
    load_scored_groups,
    make_example,
    prepare_resume_file,
    read_jsonl,
    should_swap,
)

RECORD = {
    "domain": "general",
    "language": "english",
    "context": [{"role": "user", "content": "hi"}],
    "response1": "resp one",
    "response2": "resp two",
    "overall_preference": -2,
    "individual_preference": [],
}


def test_gold_side_sign_semantics():
    assert gold_side(-3) == "1"
    assert gold_side(2) == "2"
    with pytest.raises(ValueError):
        gold_side(0)


def test_swap_is_deterministic():
    assert should_swap("x:1") == should_swap("x:1")


def test_make_example_gold_tracks_swap():
    for example_id in ("a:0", "a:1", "a:2", "a:3"):
        ex = make_example(RECORD, example_id)
        # preference favors response1; gold must point at wherever it landed
        winner_text = ex.response_a if ex.gold == "A" else ex.response_b
        assert winner_text == "resp one"
        if ex.swapped:
            assert ex.response_a == "resp two"
        else:
            assert ex.response_a == "resp one"


def test_load_examples(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(RECORD) + "\n" + json.dumps(RECORD) + "\n")
    examples = load_examples(path, "train")
    assert [e.example_id for e in examples] == ["train:0", "train:1"]
    assert examples[0].domain == "general"


def test_prepare_resume_file_removes_failures_and_duplicates(tmp_path):
    path = tmp_path / "partial.jsonl"
    records = [
        {"example_id": "train:0", "value": "old"},
        {"example_id": "train:1", "error": "temporary failure"},
        {"example_id": "train:0", "value": "new"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    done = prepare_resume_file(path, lambda record: "error" not in record)

    assert done == {"train:0"}
    assert read_jsonl(path) == [{"example_id": "train:0", "value": "new"}]


def test_prepare_resume_file_removes_truncated_last_line(tmp_path):
    path = tmp_path / "interrupted.jsonl"
    complete = {"example_id": "train:0", "value": "complete"}
    path.write_text(json.dumps(complete) + "\n" + '{"example_id": "train:1"')

    done = prepare_resume_file(path, lambda _record: True)

    assert done == {"train:0"}
    assert read_jsonl(path) == [complete]


def test_load_rollout_groups_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "rollout.jsonl"
    record = {"example_id": "train:0", "model": "m", "generation": ["x"]}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="duplicate example_id"):
        load_rollout_groups([path])


def test_load_scored_groups_checks_input_metadata(tmp_path):
    example = make_example(RECORD, "train:0")
    path = tmp_path / "scored.jsonl"
    scored = {
        "example_id": example.example_id,
        "domain": example.domain,
        "gold": example.gold,
        "judgments": [],
    }
    path.write_text(json.dumps(scored) + "\n")

    groups, stats = load_scored_groups([path], {example.example_id: example})
    assert groups == [scored]
    assert stats == {"errors": 0, "discarded": 0}

    scored["gold"] = "B" if example.gold == "A" else "A"
    path.write_text(json.dumps(scored) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        load_scored_groups([path], {example.example_id: example})
