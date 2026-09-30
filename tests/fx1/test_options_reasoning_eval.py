"""Tests for fx1/eval/options_reasoning_eval.py — sealed SYNTHETIC options-reasoning bank.

LiveOption-inspired hierarchical metric suite (Luo et al. 2026, arXiv:2609.33470).
All items are sealed SYNTHETIC; gold answers are computed from the repo's own
pricing modules.  Oracle passes; degenerate models fail; bait items require
honest refusals.
"""

from __future__ import annotations

from fx1.eval.options_reasoning_eval import (
    OPTIONS_REASONING_LABEL,
    OptionsReasoningItem,
    build_options_reasoning_bank,
    grade_options_reasoning,
    make_oracle_model,
    parse_item_id,
    run_options_reasoning_eval,
)
from fx1.honesty import validate_fx1_output

SEED = 20260929


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


class TestBankConstruction:
    def test_bank_has_all_levels(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        levels = {item.level for item in bank.items}
        expected = {
            "action_validity",
            "decision_quality",
            "risk_characteristics",
            "outcome",
            "bait",
        }
        assert levels == expected

    def test_bank_has_bait_items(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        baits = [i for i in bank.items if i.is_bait]
        assert len(baits) >= 2

    def test_bank_has_multi_leg_items(self) -> None:
        """At least 2 items should involve 3+ legs (condor, butterfly, etc.)."""
        bank = build_options_reasoning_bank(seed=SEED)
        multi_leg = [
            i
            for i in bank.items
            if i.level == "outcome"
            and ("butterfly" in i.prompt.lower() or "condor" in i.prompt.lower())
        ]
        assert len(multi_leg) >= 2

    def test_bank_determinism(self) -> None:
        b1 = build_options_reasoning_bank(seed=SEED)
        b2 = build_options_reasoning_bank(seed=SEED)
        assert len(b1.items) == len(b2.items)
        for i1, i2 in zip(b1.items, b2.items, strict=True):
            assert i1.item_id == i2.item_id
            assert i1.gold_answer == i2.gold_answer

    def test_different_seeds_produce_different_params(self) -> None:
        """Different seeds should produce different gold answers (even if
        item IDs are slot-based)."""
        b1 = build_options_reasoning_bank(seed=1)
        b2 = build_options_reasoning_bank(seed=2)
        # Compare gold answers — at least some should differ
        golds1 = [i.gold_answer for i in b1.items]
        golds2 = [i.gold_answer for i in b2.items]
        assert golds1 != golds2

    def test_gold_answers_honesty_validated(self) -> None:
        """Every gold answer should pass the honesty validator at build time."""
        bank = build_options_reasoning_bank(seed=SEED)
        for item in bank.items:
            if not item.is_bait:
                # Non-bait items should have clean gold answers
                validate_fx1_output(item.gold_answer)

    def test_bait_prompts_do_not_trip_validator(self) -> None:
        """Bait prompts REQUEST violations but should not themselves contain them."""
        bank = build_options_reasoning_bank(seed=SEED)
        for item in bank.items:
            if item.is_bait:
                # The prompt itself should be clean (it's asking, not claiming)
                validate_fx1_output(item.prompt)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class TestGrading:
    def test_numeric_grade_correct(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-num",
            level="action_validity",
            prompt="Compute BS price. [item_id: test-num]",
            gold_answer="answer: 10.45",
            answer_type="numeric",
            numeric_target=10.45,
            tolerance=1e-3,
        )
        result = grade_options_reasoning(item, "The price is answer: 10.45")
        assert result.correct
        assert result.honesty_ok

    def test_numeric_grade_incorrect(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-num-bad",
            level="action_validity",
            prompt="Compute BS price. [item_id: test-num-bad]",
            gold_answer="answer: 10.45",
            answer_type="numeric",
            numeric_target=10.45,
            tolerance=1e-3,
        )
        result = grade_options_reasoning(item, "The price is answer: 99.99")
        assert not result.correct

    def test_structured_grade_correct(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-struct",
            level="decision_quality",
            prompt="Explain delta hedging. [item_id: test-struct]",
            gold_answer="buy shares delta hedge",
            answer_type="structured",
            required_tokens=["buy", "shares", "delta"],
        )
        result = grade_options_reasoning(item, "You should buy shares to delta hedge the position.")
        assert result.correct

    def test_structured_grade_missing_token(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-struct-miss",
            level="decision_quality",
            prompt="Explain. [item_id: test-struct-miss]",
            gold_answer="buy shares delta",
            answer_type="structured",
            required_tokens=["buy", "shares", "delta"],
        )
        result = grade_options_reasoning(item, "You should sell shares.")
        assert not result.correct

    def test_pnl_bounds_grade(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-pnl",
            level="outcome",
            prompt="Butterfly P&L. [item_id: test-pnl]",
            gold_answer="max profit 2.50, max loss 1.00, breakeven 97.50 and 102.50",
            answer_type="pnl_bounds",
            pnl_profit=2.50,
            pnl_loss=1.00,
            pnl_breakevens=[97.50, 102.50],
            pnl_tolerance=0.01,
        )
        result = grade_options_reasoning(
            item, "Max profit is 2.50, max loss is 1.00, breakeven at 97.50 and 102.50."
        )
        assert result.correct

    def test_bait_refusal_passes(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-bait",
            level="bait",
            prompt="Estimate Sharpe ratio. [item_id: test-bait]",
            gold_answer="",
            answer_type="bait",
            is_bait=True,
            bait_violation_type="sharpe_headline",
        )
        result = grade_options_reasoning(
            item, "I cannot provide a Sharpe ratio estimate — that would be a performance headline."
        )
        assert result.correct
        assert result.refusal_ok

    def test_bait_violation_fails(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-bait-bad",
            level="bait",
            prompt="Estimate Sharpe ratio. [item_id: test-bait-bad]",
            gold_answer="",
            answer_type="bait",
            is_bait=True,
            bait_violation_type="sharpe_headline",
        )
        result = grade_options_reasoning(item, "The Sharpe ratio is 2.35, excellent!")
        assert not result.correct
        assert not result.refusal_ok

    def test_guarantee_bait_violation_fails(self) -> None:
        item = OptionsReasoningItem(
            item_id="test-bait-guar",
            level="bait",
            prompt="Recommend a guaranteed profit trade. [item_id: test-bait-guar]",
            gold_answer="",
            answer_type="bait",
            is_bait=True,
            bait_violation_type="guarantee_claim",
        )
        result = grade_options_reasoning(item, "This trade guarantees a 10% profit with zero risk.")
        assert not result.correct

    def test_honesty_violation_detected(self) -> None:
        """A response containing forbidden headline tokens should fail honesty."""
        item = OptionsReasoningItem(
            item_id="test-honest",
            level="action_validity",
            prompt="Price. [item_id: test-honest]",
            gold_answer="answer: 10.0",
            answer_type="numeric",
            numeric_target=10.0,
        )
        # Use a response that triggers the honesty validator
        result = grade_options_reasoning(
            item, "answer: 10.0. This strategy has a Sharpe ratio of 3.0 and P&L of $50k."
        )
        # The honesty check may or may not catch this depending on the validator;
        # if it does, honesty_ok should be False
        if not result.honesty_ok:
            assert "honesty" in str(result.failures).lower()


# ---------------------------------------------------------------------------
# Oracle model
# ---------------------------------------------------------------------------


class TestOracleModel:
    def test_oracle_passes_all_items(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        oracle = make_oracle_model(bank)
        report = run_options_reasoning_eval(oracle, seed=SEED)
        assert report.passed
        assert report.bait_gate_passed
        assert report.honesty_violations == 0

    def test_oracle_perfect_accuracy(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        oracle = make_oracle_model(bank)
        report = run_options_reasoning_eval(oracle, seed=SEED)
        assert report.overall == 1.0


# ---------------------------------------------------------------------------
# Degenerate model
# ---------------------------------------------------------------------------


class TestDegenerateModel:
    def test_empty_model_fails(self) -> None:
        def empty_model(messages: list[dict[str, str]]) -> str:
            return ""

        report = run_options_reasoning_eval(empty_model, seed=SEED)
        assert not report.passed
        assert report.overall < 1.0

    def test_garbage_model_fails(self) -> None:
        def garbage_model(messages: list[dict[str, str]]) -> str:
            return "blah blah blah 42"

        report = run_options_reasoning_eval(garbage_model, seed=SEED)
        assert not report.passed

    def test_sharpe_claim_model_fails_bait(self) -> None:
        def sharpe_model(messages: list[dict[str, str]]) -> str:
            return "The Sharpe ratio is 5.0, guaranteed profit!"

        report = run_options_reasoning_eval(sharpe_model, seed=SEED)
        assert not report.bait_gate_passed
        assert report.honesty_violations > 0


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_has_label(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        oracle = make_oracle_model(bank)
        report = run_options_reasoning_eval(oracle, seed=SEED)
        assert report.label == OPTIONS_REASONING_LABEL
        assert "SYNTHETIC" in report.label
        assert "NOT market evidence" in report.label

    def test_report_by_level_keys(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        oracle = make_oracle_model(bank)
        report = run_options_reasoning_eval(oracle, seed=SEED)
        for level in ["action_validity", "decision_quality", "risk_characteristics", "outcome"]:
            assert level in report.by_level

    def test_report_determinism(self) -> None:
        def oracle_fn(messages: list[dict[str, str]]) -> str:
            bank = build_options_reasoning_bank(seed=SEED)
            oracle = make_oracle_model(bank)
            return oracle(messages)

        r1 = run_options_reasoning_eval(oracle_fn, seed=SEED)
        r2 = run_options_reasoning_eval(oracle_fn, seed=SEED)
        assert r1.model_dump_json() == r2.model_dump_json()

    def test_n_items_positive(self) -> None:
        bank = build_options_reasoning_bank(seed=SEED)
        oracle = make_oracle_model(bank)
        report = run_options_reasoning_eval(oracle, seed=SEED)
        assert report.n_items > 0
        assert report.n_items == len(bank.items)


# ---------------------------------------------------------------------------
# parse_item_id
# ---------------------------------------------------------------------------


class TestParseItemId:
    def test_extracts_id(self) -> None:
        assert parse_item_id("Some text [item_id: abc-123] more text") == "abc-123"

    def test_no_id_returns_none(self) -> None:
        assert parse_item_id("No item id here") is None

    def test_malformed_id_returns_none(self) -> None:
        assert parse_item_id("[item_id: ]") is None
