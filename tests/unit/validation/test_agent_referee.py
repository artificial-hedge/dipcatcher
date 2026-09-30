"""Tests for validation/agent_referee.py — anytime-valid frozen referee.

Seeded SYNTHETIC planted-world tests (Qu, Chen & Wang 2026, arXiv:2609.27051).
All data is synthetic; no market evidence, no live-trading claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.validation.agent_referee import (
    CandidateSubmission,
    RefereeLedger,
    RefereeVerdict,
    frozen_referee_evalue,
    leaky_referee_contrast,
    referee_fdr_control,
)

SEED = 20260929


# ---------------------------------------------------------------------------
# frozen_referee_evalue
# ---------------------------------------------------------------------------


class TestFrozenRefereeEvalue:
    def test_null_returns_evalue_near_one(self) -> None:
        """Under the null (mean=0), the e-value should be near 1 on average."""
        rng = np.random.default_rng(SEED)
        e_values = []
        for _ in range(100):
            r = rng.standard_normal(200)
            e_values.append(frozen_referee_evalue(r))
        mean_e = np.mean(e_values)
        # Under the null, E[E] <= 1; allow some MC noise
        assert mean_e < 2.0

    def test_positive_edge_gives_large_evalue(self) -> None:
        """With a positive edge, the e-value should grow."""
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) + 0.5  # true edge = 0.5
        e_val = frozen_referee_evalue(r)
        assert e_val > 10.0  # strong evidence

    def test_negative_edge_gives_small_evalue(self) -> None:
        """With a negative edge, the e-value should be small (bust or near 0)."""
        rng = np.random.default_rng(SEED)
        r = rng.standard_normal(500) - 0.5
        e_val = frozen_referee_evalue(r)
        assert e_val < 1.0

    def test_fail_closed_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            frozen_referee_evalue(np.array([]))

    def test_fail_closed_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            frozen_referee_evalue(np.array([1.0, np.nan]))

    def test_fail_closed_bad_null_std(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            frozen_referee_evalue(np.array([1.0, 2.0]), null_std=0.0)

    def test_fail_closed_bad_bet_fraction(self) -> None:
        with pytest.raises(ValueError, match="bet_fraction"):
            frozen_referee_evalue(np.array([1.0, 2.0]), bet_fraction=1.5)

    def test_determinism(self) -> None:
        r = np.array([0.1, -0.2, 0.3, 0.15, -0.1])
        e1 = frozen_referee_evalue(r)
        e2 = frozen_referee_evalue(r)
        assert e1 == e2


# ---------------------------------------------------------------------------
# RefereeLedger
# ---------------------------------------------------------------------------


class TestRefereeLedger:
    def test_submit_and_count(self) -> None:
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=10))
        ledger.submit(CandidateSubmission("c2", submission_time=20))
        assert ledger.n_submissions == 2

    def test_monotone_submission_times(self) -> None:
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=10))
        with pytest.raises(ValueError, match="must be >="):
            ledger.submit(CandidateSubmission("c2", submission_time=5))
        # Equal times should be allowed (batch submission)
        ledger.submit(CandidateSubmission("c3", submission_time=10))

    def test_freeze_blocks_submissions(self) -> None:
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=10))
        ledger.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            ledger.submit(CandidateSubmission("c2", submission_time=20))

    def test_evaluate_post_submission_only(self) -> None:
        """The referee must only use data AFTER submission_time."""
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=5))

        # Outcome stream: first 6 are noise (times 0-5), rest have edge (times 6-24)
        rng = np.random.default_rng(SEED)
        outcomes = np.concatenate([rng.standard_normal(6), rng.standard_normal(19) + 1.0])

        verdicts = ledger.evaluate(outcomes)
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.candidate_id == "c1"
        assert v.n_post_sub_obs == 19  # times 6-24 = 19 observations
        assert v.e_value > 1.0  # should detect the edge

    def test_evaluate_no_post_sub_data(self) -> None:
        """If submission_time >= len(outcomes), e-value should be 1 (no evidence)."""
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=100))
        outcomes = np.ones(50)
        verdicts = ledger.evaluate(outcomes)
        assert verdicts[0].e_value == 1.0
        assert verdicts[0].n_post_sub_obs == 0

    def test_evaluate_with_time_index(self) -> None:
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=10))
        outcomes = np.ones(20)
        time_index = np.arange(20) * 2  # times 0, 2, 4, ..., 38
        verdicts = ledger.evaluate(outcomes, time_index=time_index)
        # Only times > 10 are used: times 12, 14, ..., 38 → 14 observations
        assert verdicts[0].n_post_sub_obs == 14


# ---------------------------------------------------------------------------
# FDR control
# ---------------------------------------------------------------------------


class TestFDRControl:
    def test_ebh_admits_strong_evalues(self) -> None:
        verdicts = [
            RefereeVerdict("c1", e_value=100.0, n_post_sub_obs=100),
            RefereeVerdict("c2", e_value=50.0, n_post_sub_obs=100),
            RefereeVerdict("c3", e_value=1.5, n_post_sub_obs=100),
        ]
        result = referee_fdr_control(verdicts, alpha=0.1)
        assert result[0].admitted
        assert result[1].admitted
        assert not result[2].admitted

    def test_ebh_rejects_all_weak(self) -> None:
        verdicts = [
            RefereeVerdict("c1", e_value=1.1, n_post_sub_obs=100),
            RefereeVerdict("c2", e_value=1.2, n_post_sub_obs=100),
        ]
        result = referee_fdr_control(verdicts, alpha=0.1)
        assert not result[0].admitted
        assert not result[1].admitted

    def test_ebh_empty(self) -> None:
        result = referee_fdr_control([], alpha=0.1)
        assert result == []


# ---------------------------------------------------------------------------
# Leaky referee contrast (red-team)
# ---------------------------------------------------------------------------


class TestLeakyRefereeContrast:
    def test_frozen_admits_fewer_noise_than_leaky(self) -> None:
        """The frozen referee should admit fewer noise factors than the leaky one."""
        result = leaky_referee_contrast(
            n_true=5,
            n_noise=20,
            n_periods=200,
            true_edge=0.5,
            seed=SEED,
            alpha=0.1,
            submission_time=50,
        )
        # The leaky referee sees pre-submission noise and may admit spurious factors
        assert result["leaky_noise_admitted"] >= result["frozen_noise_admitted"]

    def test_frozen_admits_true_factors(self) -> None:
        """The frozen referee should admit true factors (with enough data)."""
        result = leaky_referee_contrast(
            n_true=5,
            n_noise=10,
            n_periods=500,
            true_edge=1.0,
            seed=SEED,
            alpha=0.1,
            submission_time=50,
        )
        assert result["frozen_true_admitted"] > 0

    def test_determinism(self) -> None:
        r1 = leaky_referee_contrast(n_true=3, n_noise=5, n_periods=100, seed=SEED)
        r2 = leaky_referee_contrast(n_true=3, n_noise=5, n_periods=100, seed=SEED)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Integration: referee_evaluate
# ---------------------------------------------------------------------------


class TestRefereeEvaluate:
    def test_end_to_end(self) -> None:
        ledger = RefereeLedger(alpha=0.1)
        ledger.submit(CandidateSubmission("true_factor", submission_time=10))
        ledger.submit(CandidateSubmission("noise_factor", submission_time=10))

        rng = np.random.default_rng(SEED)
        # True factor has edge, noise doesn't
        outcomes_true = rng.standard_normal(100) + 0.8
        outcomes_noise = rng.standard_normal(100)

        # Evaluate true factor
        verdicts_true = ledger.evaluate(outcomes_true)
        # Evaluate noise factor (same ledger, different stream)
        verdicts_noise = ledger.evaluate(outcomes_noise)

        # True factor should have higher e-value
        assert verdicts_true[0].e_value > verdicts_noise[0].e_value
