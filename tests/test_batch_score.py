import json

import pytest

from decert.batch_score import (
    assemble_groups,
    phase1_requests,
    phase2_requests,
    read_batch_texts,
    required_phase1_ids,
    required_phase2_ids,
    write_sharded_requests,
)
from decert.llm import FakeClient
from decert.pipeline import Scorer
from tests.test_pipeline import CORRECT_JUDGMENT, EXAMPLE, WRONG_JUDGMENT, fake_verifier

GENERATIONS = [
    {"model": "m1", "text": CORRECT_JUDGMENT},
    {"model": "m2", "text": WRONG_JUDGMENT},
    {"model": "m2", "text": "not a judgment"},
]
GROUPS = {"t:0": GENERATIONS}
EXAMPLES = {"t:0": EXAMPLE}


def answer_requests(requests):
    """Simulate a completed batch: answer every request with fake_verifier."""
    return {
        req["custom_id"]: fake_verifier(req["body"]["messages"]) for req in requests
    }


def test_request_structure_and_skips():
    reqs = phase1_requests(EXAMPLES, GROUPS, "gpt-4o")
    ids = [r["custom_id"] for r in reqs]
    # analyzer + (grounding + validity) for the 2 parseable judgments only
    assert ids == ["t:0|analyzer", "t:0|g0", "t:0|v0", "t:0|g1", "t:0|v1"]
    assert reqs[0]["body"]["model"] == "gpt-4o"
    assert reqs[0]["body"]["temperature"] == 0.0

    split_models = phase1_requests(
        EXAMPLES, GROUPS, "Qwen/Qwen3-14B", analyzer_model="gpt-4o"
    )
    assert split_models[0]["body"]["model"] == "gpt-4o"
    assert all(
        request["body"]["model"] == "Qwen/Qwen3-14B"
        for request in split_models[1:]
    )

    phase1_texts = answer_requests(reqs)
    reqs2 = phase2_requests(EXAMPLES, GROUPS, phase1_texts, "gpt-4o")
    assert [r["custom_id"] for r in reqs2] == ["t:0|m0", "t:0|m1"]
    assert required_phase1_ids(EXAMPLES, GROUPS) == set(ids)
    assert required_phase2_ids(EXAMPLES, GROUPS, phase1_texts) == {
        "t:0|m0",
        "t:0|m1",
    }


def test_batch_equals_sync_scoring():
    sync = Scorer(client=FakeClient(handler=fake_verifier)).score_group(
        EXAMPLE, GENERATIONS
    )

    p1 = phase1_requests(EXAMPLES, GROUPS, "gpt-4o")
    phase1_texts = answer_requests(p1)
    p2 = phase2_requests(EXAMPLES, GROUPS, phase1_texts, "gpt-4o")
    phase2_texts = answer_requests(p2)
    batch = assemble_groups(EXAMPLES, GROUPS, phase1_texts, phase2_texts)

    assert len(batch) == 1
    assert batch[0] == sync


def test_missing_batch_results_are_pessimistic():
    p1 = phase1_requests(EXAMPLES, GROUPS, "gpt-4o")
    phase1_texts = answer_requests(p1)
    # matcher results entirely missing (e.g. errored batch requests)
    records = assemble_groups(EXAMPLES, GROUPS, phase1_texts, {})
    good = records[0]["judgments"][0]
    assert good["parse_ok"] and good["s_cov"] == 0.0
    # analyzer missing too: the entire group is discarded
    records = assemble_groups(EXAMPLES, GROUPS, {}, {})
    assert records[0]["discarded"] is True
    assert records[0]["discard_reason"] == "empty_issue_set"


def test_read_batch_texts_skips_errors(tmp_path):
    lines = [
        {
            "custom_id": "a|analyzer",
            "response": {"body": {"choices": [{"message": {"content": "[]"}}]}},
            "error": None,
        },
        {"custom_id": "a|g0", "response": None, "error": {"message": "boom"}},
    ]
    path = tmp_path / "out.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    assert read_batch_texts([path]) == {"a|analyzer": "[]"}


def test_read_batch_texts_rejects_duplicate_results(tmp_path):
    line = {
        "custom_id": "a|analyzer",
        "response": {"body": {"choices": [{"message": {"content": "[]"}}]}},
        "error": None,
    }
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(line) + "\n" + json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="duplicate batch result"):
        read_batch_texts([path])


def test_write_sharded_requests(tmp_path):
    reqs = [{"custom_id": f"x{i}", "body": {}} for i in range(5)]
    paths = write_sharded_requests(tmp_path, "phase1", reqs)
    assert [p.name for p in paths] == ["phase1_000.jsonl"]
    written = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert written == reqs
