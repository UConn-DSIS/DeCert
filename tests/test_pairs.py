import pytest

from decert.pairs import (
    build_maxmin_pairs,
    build_sft_targets,
    process_score,
)
from decert.types import PairExample

EXAMPLE = PairExample(
    example_id="t:0",
    domain="general",
    language="english",
    context=[{"role": "user", "content": "hi"}],
    response_a="a",
    response_b="b",
    gold="A",
    swapped=False,
)


def judgment(outcome, q_proc, text, model="m", parse_ok=True):
    lam = 0.5
    return {
        "model": model,
        "text": text,
        "parse_ok": parse_ok,
        "outcome": outcome,
        "s_snd": q_proc,
        "s_cov": q_proc,
        "s_dec": q_proc,
        "q_proc": q_proc,
        "q_total": (outcome + lam * q_proc) / (1 + lam),
    }


def group(judgments, example_id="t:0"):
    return {
        "example_id": example_id,
        "domain": "general",
        "gold": "A",
        "issues": [],
        "judgments": judgments,
    }


def test_sft_targets_decoupled_from_outcome_contrast():
    groups = [
        # all-correct group: no outcome contrast, but a valid SFT source
        group([judgment(1, 0.9, "best"), judgment(1, 0.6, "ok")], "t:0"),
    ]
    records, stats = build_sft_targets(groups, {"t:0": EXAMPLE}, min_qproc=0.7)
    assert stats["kept"] == 1
    assert records[0]["target"] == "best"
    assert records[0]["target_model"] == "m"
    assert records[0]["target_scores"]["outcome"] == 1


def test_sft_targets_quality_floor_and_no_correct():
    low_quality = group([judgment(1, 0.5, "meh"), judgment(0, 0.9, "wrong")], "t:0")
    all_wrong = group([judgment(0, 0.9, "a"), judgment(0, 0.8, "b")], "t:0")
    records, stats = build_sft_targets(
        [low_quality, all_wrong], {"t:0": EXAMPLE}, min_qproc=0.7
    )
    assert records == []
    assert stats["below_min_qproc"] == 1
    assert stats["no_correct"] == 1
    # floor off -> the low-quality group qualifies
    records, stats = build_sft_targets([low_quality], {"t:0": EXAMPLE}, min_qproc=0.0)
    assert stats["kept"] == 1 and records[0]["target"] == "meh"


def test_maxmin_mixed_group_reduces_to_outcome_pair():
    g = group(
        [
            judgment(1, 0.4, "correct-low"),
            judgment(0, 0.95, "wrong-high"),
            judgment(1, 0.9, "correct-high"),
            judgment(0, 0.1, "wrong-low"),
        ]
    )
    records, stats = build_maxmin_pairs([g], {"t:0": EXAMPLE})
    assert stats["kept"] == 1
    pair = records[0]
    assert pair["pair_type"] == "maxmin"
    # Q_total dominance: chosen is the best CORRECT one even though
    # wrong-high has the highest q_proc
    assert pair["chosen"] == "correct-high"
    assert pair["rejected"] == "wrong-low"


def test_maxmin_uses_all_correct_groups():
    g = group([judgment(1, 0.9, "best"), judgment(1, 0.5, "worst")])
    records, stats = build_maxmin_pairs([g], {"t:0": EXAMPLE})
    assert stats["kept"] == 1
    pair = records[0]
    assert pair["chosen"] == "best" and pair["rejected"] == "worst"
    # same-verdict pair: both sides correct
    assert pair["chosen_scores"]["outcome"] == 1
    assert pair["rejected_scores"]["outcome"] == 1


def test_maxmin_arithmetic_aggregation_can_change_pair_order():
    spiky = judgment(1, 0.01 ** (1 / 3), "spiky")
    spiky.update({"s_snd": 1.0, "s_cov": 1.0, "s_dec": 0.01})
    balanced = judgment(1, 0.4, "balanced")
    candidates = group([spiky, balanced])

    geometric, _ = build_maxmin_pairs(
        [candidates], {"t:0": EXAMPLE}, same_verdict_delta=0.0
    )
    arithmetic, _ = build_maxmin_pairs(
        [candidates],
        {"t:0": EXAMPLE},
        same_verdict_delta=0.0,
        process_aggregation="arithmetic",
    )

    assert geometric[0]["chosen"] == "balanced"
    assert arithmetic[0]["chosen"] == "spiky"
    assert arithmetic[0]["process_aggregation"] == "arithmetic"
    assert arithmetic[0]["chosen_scores"]["process_score"] == pytest.approx(0.67)
    assert process_score(spiky, "arithmetic") == pytest.approx(0.67)


def test_maxmin_skips_all_wrong_and_small_groups():
    all_wrong = group([judgment(0, 0.9, "a"), judgment(0, 0.2, "b")])
    single = group([judgment(1, 0.9, "a"), judgment(1, 0.8, "b", parse_ok=False)])
    records, stats = build_maxmin_pairs([all_wrong, single], {"t:0": EXAMPLE})
    assert records == []
    assert stats["all_wrong"] == 1
    assert stats["fewer_than_two_parseable"] == 1


def test_maxmin_same_verdict_delta_only_gates_all_correct_groups():
    tied = group([judgment(1, 0.7, "a"), judgment(1, 0.7, "b")])
    narrow = group([judgment(1, 0.90, "a"), judgment(1, 0.70, "b")])
    # default 0.25: both same-verdict pairs sit below the noise floor
    records, stats = build_maxmin_pairs([tied, narrow], {"t:0": EXAMPLE})
    assert records == [] and stats["same_verdict_below_delta"] == 2
    # a smaller delta lets the 0.20-gap pair through
    records, stats = build_maxmin_pairs(
        [narrow], {"t:0": EXAMPLE}, same_verdict_delta=0.15
    )
    assert stats["kept"] == 1
    # mixed pairs are exempt: tiny certificate gap, but outcomes differ
    mixed = group([judgment(1, 0.41, "correct"), judgment(0, 0.40, "wrong")])
    records, stats = build_maxmin_pairs(
        [mixed], {"t:0": EXAMPLE}, same_verdict_delta=0.5
    )
    assert stats["kept"] == 1
    assert records[0]["chosen"] == "correct" and records[0]["rejected"] == "wrong"


def test_discarded_group_never_produces_training_data():
    discarded = {
        "example_id": "t:0",
        "domain": "general",
        "gold": "A",
        "discarded": True,
        "discard_reason": "empty_issue_set",
    }
    sft, _ = build_sft_targets([discarded], {"t:0": EXAMPLE})
    dpo, _ = build_maxmin_pairs([discarded], {"t:0": EXAMPLE})
    assert sft == []
    assert dpo == []
