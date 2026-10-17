from decert.parsing import is_parseable, parse_judgment, validation_errors
from decert.rubrics import Criterion

RUBRIC = [
    Criterion("correctness", 0.8, "Correctness"),
    Criterion("clarity", 0.2, "Clarity"),
]

GOOD = """\
<comparison>
<criterion name="correctness">
<obs side="A">A says "2+2=4".</obs>
<obs side="B">B says "2+2=5".</obs>
<result winner="A">A is arithmetically correct while B is not.</result>
</criterion>
<criterion name="clarity">
<obs side="A">A gives a short direct answer.</obs>
<obs side="B">B uses clear numbered steps.</obs>
<result winner="B">B is better organized.</result>
</criterion>
</comparison>
<decision>Correctness dominates clarity, so A wins.</decision>
<answer>[[A]]</answer>
"""


def test_parse_full_judgment():
    j = parse_judgment(GOOD)
    assert j.verdict == "A"
    assert [c.name for c in j.criteria] == ["correctness", "clarity"]
    assert j.criteria[0].winner == "A"
    assert j.criteria[1].winner == "B"
    assert len(j.criteria[0].observations) == 2
    assert j.criteria[0].observations[1].side == "B"
    assert "dominates" in j.decision_text
    assert is_parseable(j)
    assert is_parseable(j, RUBRIC)


def test_tie_winner_and_case_insensitivity():
    text = GOOD.replace('winner="B"', 'winner="tie"').replace("[[A]]", "[[b]]")
    j = parse_judgment(text)
    assert j.criteria[1].winner == "tie"
    assert j.verdict == "B"


def test_missing_answer_gives_none():
    j = parse_judgment(GOOD.replace("<answer>[[A]]</answer>", ""))
    assert j.verdict is None
    assert not is_parseable(j)


def test_malformed_block_is_dropped_not_fatal():
    broken = GOOD.replace('<result winner="B">B is better organized.</result>', "")
    j = parse_judgment(broken)
    assert [c.name for c in j.criteria] == ["correctness"]
    assert not is_parseable(j, RUBRIC)


def test_garbage_input():
    j = parse_judgment("I think A is better.")
    assert j.verdict is None
    assert j.criteria == []


def test_think_block_is_stripped_before_parsing():
    # a draft answer tag inside <think> must never become the verdict
    text = "<think>Hmm, maybe <answer>[[B]]</answer>? No wait...</think>\n" + GOOD
    j = parse_judgment(text)
    assert j.verdict == "A"
    assert [c.name for c in j.criteria] == ["correctness", "clarity"]
    assert j.raw == text  # raw keeps the original output


def test_truncated_think_block_is_unparseable():
    truncated = "<think>reasoning that ran out of tokens " + GOOD
    j = parse_judgment(truncated)
    assert j.verdict is None
    assert j.criteria == []


def test_policy_validation_rejects_missing_unknown_and_duplicate_criteria():
    missing = parse_judgment(
        GOOD.replace(
            GOOD[
                GOOD.index('<criterion name="clarity">') : GOOD.index("</comparison>")
            ],
            "",
        )
    )
    assert "missing policy criteria: clarity" in validation_errors(missing, RUBRIC)

    unknown = parse_judgment(GOOD.replace('name="clarity"', 'name="style"'))
    errors = validation_errors(unknown, RUBRIC)
    assert "criteria outside the policy: style" in errors
    assert "missing policy criteria: clarity" in errors

    clarity_block = GOOD[
        GOOD.index('<criterion name="clarity">') : GOOD.index("</comparison>")
    ]
    duplicate = parse_judgment(
        GOOD.replace("</comparison>", clarity_block + "</comparison>")
    )
    assert "duplicate criteria: clarity" in validation_errors(duplicate, RUBRIC)


def test_policy_validation_requires_both_sides_and_decision():
    no_a = parse_judgment(
        GOOD.replace('<obs side="A">A gives a short direct answer.</obs>\n', "")
    )
    assert any("both A and B" in error for error in validation_errors(no_a, RUBRIC))

    no_decision = parse_judgment(
        GOOD.replace(
            "<decision>Correctness dominates clarity, so A wins.</decision>", ""
        )
    )
    assert any(
        "<decision>" in error for error in validation_errors(no_decision, RUBRIC)
    )


def test_criterion_outside_comparison_does_not_satisfy_policy():
    clarity_start = GOOD.index('<criterion name="clarity">')
    comparison_end = GOOD.index("</comparison>")
    clarity_block = GOOD[clarity_start:comparison_end]
    text = GOOD.replace(clarity_block, "").replace(
        "</comparison>", "</comparison>\n" + clarity_block
    )
    judgment = parse_judgment(text)
    assert [block.name for block in judgment.criteria] == ["correctness"]
    assert "missing policy criteria: clarity" in validation_errors(judgment, RUBRIC)


def test_policy_validation_rejects_extra_result_and_malformed_tags():
    extra_result = parse_judgment(
        GOOD.replace(
            '<result winner="A">A is arithmetically correct while B is not.</result>',
            '<result winner="A">First result.</result>'
            '<result winner="B">Second result.</result>',
        )
    )
    assert any("exactly one" in error for error in validation_errors(extra_result, RUBRIC))

    malformed = parse_judgment(
        GOOD.replace(
            "</comparison>",
            '<criterion name="style"><obs side="A">Unclosed block</obs>'
            "</comparison>",
        )
    )
    assert "malformed or unclosed <criterion> block" in validation_errors(
        malformed, RUBRIC
    )


def test_validation_rejects_extra_malformed_top_level_tag():
    judgment = parse_judgment(GOOD + "\n<answer>free-form answer</answer>")
    assert "expected exactly one <answer> containing [[A]] or [[B]]" in (
        validation_errors(judgment, RUBRIC)
    )
