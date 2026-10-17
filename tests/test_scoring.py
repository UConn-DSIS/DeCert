import math

import pytest

from decert.parsing import parse_judgment
from decert.scoring import (
    claim_reliabilities,
    coverage,
    criterion_intervals,
    criterion_qualities,
    decision_support,
    maximum_weight_matching,
    q_proc,
    q_total,
    soundness,
)
from decert.types import MaterialIssue

WEIGHTS = {"correctness": 0.8, "clarity": 0.2}

JUDGMENT = parse_judgment(
    """\
<comparison>
<criterion name="correctness">
<obs side="A">A's final answer is right.</obs>
<obs side="B">B divides by a possibly-zero variable.</obs>
<result winner="A">A is correct, B has a critical error.</result>
</criterion>
<criterion name="clarity">
<obs side="B">B uses numbered steps.</obs>
<result winner="B">B is clearer.</result>
</criterion>
</comparison>
<decision>Correctness dominates.</decision>
<answer>[[A]]</answer>
"""
)

PERFECT_G = {1: 1.0, 2: 1.0, 3: 1.0}
PERFECT_L = {1: 1.0, 2: 1.0}


def test_claim_reliabilities_propagate_parent_failure():
    grounding = {1: 1.0, 2: 0.0, 3: 1.0}
    claim_q = claim_reliabilities(JUDGMENT, grounding, PERFECT_L)
    # Claims 1/2 are observations; claim 3 is their criterion conclusion.
    assert claim_q == {1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}
    assert criterion_qualities(JUDGMENT, grounding, PERFECT_L) == [0.0, 1.0]


def test_soundness_perfect_claims():
    claim_q = claim_reliabilities(JUDGMENT, PERFECT_G, PERFECT_L)
    assert soundness(JUDGMENT, claim_q, WEIGHTS) == 1.0


def test_soundness_uses_criterion_bottleneck():
    fabricated = {1: 1.0, 2: 0.0, 3: 1.0}
    claim_q = claim_reliabilities(JUDGMENT, fabricated, PERFECT_L)
    assert soundness(JUDGMENT, claim_q, WEIGHTS) == 0.0


def test_soundness_missing_scores_default_to_zero():
    claim_q = claim_reliabilities(JUDGMENT, {}, {})
    assert soundness(JUDGMENT, claim_q, WEIGHTS) == 0.0


def test_maximum_weight_matching_is_global_and_one_to_one():
    issues = [
        MaterialIssue("issue one", "correctness", 1.0, "A"),
        MaterialIssue("issue two", "correctness", 1.0, "A"),
    ]
    # Issue 1 can use either claim, while issue 2 can only use claim 1. The
    # global optimum reserves claim 1 for issue 2.
    pair_scores = [[1.0, 1.0], [1.0, 0.0]]
    matches = maximum_weight_matching(issues, pair_scores, {1: 1.0, 2: 1.0})
    assert matches == [(1, 2, 1.0), (2, 1, 1.0)]


def test_matching_and_coverage_use_claim_reliability():
    issues = [
        MaterialIssue("B has math error", "correctness", 3.0, "A"),
        MaterialIssue("B clearer layout", "clarity", 1.0, "B"),
    ]
    claim_q = {1: 1.0, 2: 0.5}
    pair_scores = [[0.0, 1.0], [1.0, 0.0]]
    matches = maximum_weight_matching(issues, pair_scores, claim_q)
    assert matches == [(1, 2, 1.0), (2, 1, 1.0)]
    assert math.isclose(coverage(issues, matches, claim_q), 2.5 / 4.0)
    with pytest.raises(ValueError, match="empty material-issue set"):
        coverage([], [], {})


def test_decision_support_closed_form_trade_off():
    claim_q = claim_reliabilities(JUDGMENT, PERFECT_G, PERFECT_L)
    intervals = criterion_intervals(JUDGMENT, claim_q, WEIGHTS)
    assert intervals == {"correctness": (1.0, 1.0), "clarity": (-1.0, -1.0)}
    assert math.isclose(decision_support(JUDGMENT, intervals, WEIGHTS), 0.8)


def test_decision_support_aggregation_reversal():
    flipped = parse_judgment(JUDGMENT.raw.replace("[[A]]", "[[B]]"))
    claim_q = claim_reliabilities(flipped, PERFECT_G, PERFECT_L)
    intervals = criterion_intervals(flipped, claim_q, WEIGHTS)
    assert math.isclose(decision_support(flipped, intervals, WEIGHTS), 0.2)


def test_unverified_criterion_retains_full_uncertainty():
    grounding = {1: 1.0, 2: 0.5, 3: 1.0}
    claim_q = claim_reliabilities(JUDGMENT, grounding, PERFECT_L)
    intervals = criterion_intervals(JUDGMENT, claim_q, WEIGHTS, tau_loc=1.0)
    assert intervals["correctness"] == (-1.0, 1.0)
    assert intervals["clarity"] == (-1.0, -1.0)
    assert decision_support(JUDGMENT, intervals, WEIGHTS) == 0.0


def test_local_validity_threshold_must_be_positive():
    claim_q = claim_reliabilities(JUDGMENT, PERFECT_G, PERFECT_L)
    with pytest.raises(ValueError, match="tau_loc"):
        criterion_intervals(JUDGMENT, claim_q, WEIGHTS, tau_loc=0.0)


def test_q_proc_geometric_conjunction():
    assert q_proc(1.0, 1.0, 0.0) == 0.0
    assert math.isclose(q_proc(1.0, 1.0, 1.0), 1.0)
    assert math.isclose(q_proc(0.8, 0.5, 0.9), (0.8 * 0.5 * 0.9) ** (1 / 3))


def test_q_total_outcome_dominance():
    assert q_total(1, 0.0) > q_total(0, 1.0)
    assert math.isclose(q_total(1, 0.0, lam=0.5), 1 / 1.5)
    assert math.isclose(q_total(0, 1.0, lam=0.5), 0.5 / 1.5)
