import json

from decert.llm import FakeClient
from decert.parsing import parse_judgment
from decert.rubrics import get_rubric
from decert.types import MaterialIssue, PairExample
from decert.verify import (
    extract_json_array,
    generate_issues,
    match_issues,
    parse_issues_response,
    parse_match_response,
    score_grounding,
    score_validity,
)

EXAMPLE = PairExample(
    example_id="t:0",
    domain="general",
    language="english",
    context=[{"role": "user", "content": "What is 2+2?"}],
    response_a="2+2=4",
    response_b="2+2=5",
    gold="A",
    swapped=False,
)

JUDGMENT = parse_judgment(
    """\
<comparison>
<criterion name="correctness">
<obs side="A">A answers 4.</obs>
<obs side="B">B answers 5.</obs>
<result winner="A">A is right.</result>
</criterion>
</comparison>
<decision>A wins.</decision>
<answer>[[A]]</answer>
"""
)


def test_extract_json_array_with_fences_and_prose():
    text = 'Sure!\n```json\n[{"claim": 1, "supported": "yes"}]\n```\nDone.'
    assert extract_json_array(text) == [{"claim": 1, "supported": "yes"}]
    assert extract_json_array("no json here") == []
    assert extract_json_array("[not valid") == []
    assert extract_json_array("draft [broken], final: [1, 2]") == [1, 2]
    assert extract_json_array("[1]\nextra diagnostic: [2]") == [1]


def test_score_grounding_maps_labels_and_ignores_bad_indices():
    client = FakeClient(
        responses=[
            json.dumps(
                [
                    {"claim": 1, "supported": "yes"},
                    {"claim": 2, "supported": "partial"},
                    {"claim": 99, "supported": "yes"},
                ]
            )
        ]
    )
    scores = score_grounding(client, EXAMPLE, JUDGMENT)
    assert scores == {1: 1.0, 2: 0.5}


def test_score_validity():
    client = FakeClient(responses=[json.dumps([{"block": 1, "follows": "yes"}])])
    scores = score_validity(client, EXAMPLE, get_rubric("general"), JUDGMENT)
    assert scores == {1: 1.0}


def test_generate_issues_clamps_and_caps():
    payload = [
        {
            "issue": f"issue {i}",
            "criterion": "correctness",
            "importance": 9,
            "favors": "A",
        }
        for i in range(8)
    ]
    client = FakeClient(responses=[json.dumps(payload)])
    issues = generate_issues(client, EXAMPLE, get_rubric("general"))
    assert len(issues) == 6
    assert all(i.importance == 3.0 for i in issues)


def test_issue_parser_rejects_unknown_criteria_and_invalid_directions():
    payload = [
        {
            "issue": "unknown policy dimension",
            "criterion": "style",
            "importance": 2,
            "favors": "A",
        },
        {
            "issue": "direction is missing",
            "criterion": "correctness",
            "importance": 2,
            "favors": "tie",
        },
    ]
    assert parse_issues_response(json.dumps(payload), get_rubric("general")) == []


def test_matcher_returns_dense_pair_scores():
    issues = [
        MaterialIssue("big issue", "correctness", 3.0, "A"),
        MaterialIssue("small issue", "clarity", 1.0, "B"),
    ]
    n_claims = 3
    client = FakeClient(
        responses=[
            json.dumps(
                [
                    {"issue": 1, "scores": ["none", "full", "partial"]},
                    {"issue": 2, "scores": ["partial", "none", "full"]},
                ]
            )
        ]
    )
    scores = match_issues(client, issues, JUDGMENT)
    assert len(scores[0]) == n_claims
    assert scores == [[0.0, 1.0, 0.5], [0.5, 0.0, 1.0]]


def test_match_parser_accepts_legacy_sparse_rows():
    issues = [MaterialIssue("x", "correctness", 2.0, "A")]
    text = json.dumps([{"issue": 1, "claim": 2, "match": "partial"}])
    assert parse_match_response(text, issues, JUDGMENT) == [[0.0, 0.5, 0.0]]


def test_match_issues_empty_inputs():
    issues = [MaterialIssue("x", "correctness", 2.0, "A")]
    empty_judgment = parse_judgment("<answer>[[A]]</answer>")
    assert match_issues(FakeClient(), issues, empty_judgment) == [[]]
