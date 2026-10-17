import json

from decert.llm import FakeClient
from decert.pipeline import Scorer
from decert.types import PairExample

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

CORRECT_JUDGMENT = """\
<comparison>
<criterion name="correctness">
<obs side="A">A answers 4.</obs>
<obs side="B">B answers 5.</obs>
<result winner="A">A is right, B is wrong.</result>
</criterion>
<criterion name="instruction_following">
<obs side="A">A directly answers the question.</obs>
<obs side="B">B gives an incorrect answer to the question.</obs>
<result winner="A">A follows the request more successfully.</result>
</criterion>
<criterion name="completeness">
<obs side="A">A gives the requested correct answer.</obs>
<obs side="B">B does not give the requested correct answer.</obs>
<result winner="A">A more completely fulfills the request.</result>
</criterion>
<criterion name="clarity">
<obs side="A">A is concise and clear.</obs>
<obs side="B">B is concise but misleading because its answer is wrong.</obs>
<result winner="A">A communicates the answer more clearly.</result>
</criterion>
</comparison>
<decision>A wins on correctness.</decision>
<answer>[[A]]</answer>
"""

WRONG_JUDGMENT = CORRECT_JUDGMENT.replace("[[A]]", "[[B]]")


def fake_verifier(messages):
    system = messages[0]["content"]
    if "MATERIAL ISSUES" in system:
        return json.dumps(
            [
                {
                    "issue": "B's arithmetic is wrong",
                    "criterion": "correctness",
                    "importance": 3,
                    "favors": "A",
                }
            ]
        )
    if "audit claims" in system:
        return json.dumps([{"claim": i, "supported": "yes"} for i in range(1, 9)])
    if "criterion-level conclusions" in system:
        return json.dumps([{"block": i, "follows": "yes"} for i in range(1, 5)])
    if "every claim in a judgment covers every material issue" in system:
        scores = ["none"] * 12
        scores[2] = "full"
        return json.dumps([{"issue": 1, "scores": scores}])
    raise AssertionError(f"unexpected prompt: {system[:60]}")


def test_score_group_end_to_end():
    scorer = Scorer(client=FakeClient(handler=fake_verifier))
    group = scorer.score_group(
        EXAMPLE,
        [
            {"model": "m1", "text": CORRECT_JUDGMENT},
            {"model": "m2", "text": WRONG_JUDGMENT},
            {"model": "m2", "text": "not a judgment"},
        ],
    )
    assert group["example_id"] == "t:0"
    assert len(group["issues"]) == 1
    good, bad, unparsed = group["judgments"]

    assert good["parse_ok"] and good["outcome"] == 1
    assert good["s_snd"] > 0.99 and good["s_cov"] == 1.0 and good["s_dec"] > 0.99
    assert good["q_total"] > 0.9

    # same certificate but flipped verdict: outcome 0 and S_dec collapses
    assert bad["parse_ok"] and bad["outcome"] == 0
    assert bad["s_dec"] < 0.01
    assert bad["q_total"] < good["q_total"]

    assert not unparsed["parse_ok"]
    assert unparsed["outcome"] == 0 and unparsed["q_total"] == 0.0


def test_score_group_reuses_provided_issues():
    from decert.types import MaterialIssue

    def no_analyzer(messages):
        assert "MATERIAL ISSUES" not in messages[0]["content"], (
            "analyzer must be skipped"
        )
        return fake_verifier(messages)

    issues = [MaterialIssue("B's arithmetic is wrong", "correctness", 3.0, "A")]
    scorer = Scorer(client=FakeClient(handler=no_analyzer))
    group = scorer.score_group(
        EXAMPLE, [{"model": "m1", "text": CORRECT_JUDGMENT}], issues=issues
    )
    assert group["issues"][0]["description"] == "B's arithmetic is wrong"
    assert group["judgments"][0]["s_cov"] == 1.0


def test_matcher_client_routing():
    main = FakeClient(handler=fake_verifier)
    matcher = FakeClient(handler=fake_verifier)
    scorer = Scorer(client=main, matcher_client=matcher)
    group = scorer.score_group(EXAMPLE, [{"model": "m1", "text": CORRECT_JUDGMENT}])
    # matcher client got exactly the matcher calls, main client the rest
    assert len(matcher.calls) == 1
    assert (
        "every claim in a judgment covers every material issue"
        in (matcher.calls[0][0]["content"])
    )
    assert len(main.calls) == 3  # analyzer + grounding + validity
    assert group["judgments"][0]["s_cov"] == 1.0


def test_analyzer_client_routing():
    analyzer = FakeClient(handler=fake_verifier)
    verifier = FakeClient(handler=fake_verifier)
    matcher = FakeClient(handler=fake_verifier)
    scorer = Scorer(
        client=verifier,
        analyzer_client=analyzer,
        matcher_client=matcher,
    )

    group = scorer.score_group(
        EXAMPLE, [{"model": "m1", "text": CORRECT_JUDGMENT}]
    )

    assert len(analyzer.calls) == 1
    assert "MATERIAL ISSUES" in analyzer.calls[0][0]["content"]
    assert len(verifier.calls) == 2  # grounding + validity
    assert len(matcher.calls) == 1
    assert group["judgments"][0]["s_cov"] == 1.0


def test_empty_analyzer_issue_set_discards_group():
    def empty_analyzer(messages):
        assert "MATERIAL ISSUES" in messages[0]["content"]
        return "[]"

    client = FakeClient(handler=empty_analyzer)
    group = Scorer(client=client).score_group(
        EXAMPLE, [{"model": "m1", "text": CORRECT_JUDGMENT}]
    )
    assert group["discarded"] is True
    assert group["discard_reason"] == "empty_issue_set"
    assert len(client.calls) == 1
