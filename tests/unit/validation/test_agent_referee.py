"""Tests for validation/agent_referee.py — anytime-valid frozen referee.

Seeded SYNTHETIC planted-world tests (Qu, Chen & Wang 2026, arXiv:2609.27051).
All data is synthetic; no market evidence, no live-trading claims.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from quant_fund.validation.agent_referee import (
    CandidateSubmission,
    RefereeLedger,
    RefereeVerdict,
    frozen_referee_evalue,
    leaky_referee_contrast,
    referee_evaluate,
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

    def test_single_observation_exact_evalue(self) -> None:
        """Pin the defaults exactly: null_mean=0, null_std=1, bet_fraction=0.1.

        One observation r=0.5 gives e = 1 + 0.1 * 0.5 = 1.05 exactly.  A
        drifted default null_std (1.0 -> 2.0) would give 1.025, and a drifted
        wealth-accumulator init would scale the result by e^1.  Also guards
        the size check: a single-element input is valid, not "empty".
        """
        assert frozen_referee_evalue(np.array([0.5])) == pytest.approx(1.05, abs=1e-12)

    def test_zero_return_keeps_wealth_exactly_one(self) -> None:
        """A null-neutral observation multiplies wealth by exactly 1."""
        assert frozen_referee_evalue(np.array([0.0])) == pytest.approx(1.0, abs=1e-12)

    def test_bust_returns_exactly_zero(self) -> None:
        """factor = 1 + 0.1 * (-20) = -1 < 0 -> bust; the wealth floor is 0.0.

        A mutant returning 1.0 on bust (instead of 0.0) fails this pin; the
        rng-based negative-edge test above never busts, so it cannot.
        """
        assert frozen_referee_evalue(np.array([-20.0])) == 0.0

    def test_factor_exactly_zero_is_bust_not_domain_error(self) -> None:
        """Boundary: factor = 1 + 0.1 * (-10) = 0.0 exactly -> bust to 0.0.

        The truncation test is `factor <= 0` (inclusive).  A strict `<` mutant
        falls through to math.log(0.0) and raises a domain error instead of
        returning 0.0.
        """
        assert frozen_referee_evalue(np.array([-10.0])) == 0.0

    def test_bet_fraction_zero_rejected(self) -> None:
        """Lower bound is exclusive (0 < bet_fraction): 0.0 must raise.

        A `0 <= bet_fraction` mutant accepts 0.0 (degenerate no-betting).
        """
        with pytest.raises(ValueError, match="bet_fraction"):
            frozen_referee_evalue(np.array([1.0, 2.0]), bet_fraction=0.0)

    def test_bet_fraction_one_rejected(self) -> None:
        """Upper bound is exclusive (bet_fraction < 1): 1.0 must raise.

        A `bet_fraction <= 1` mutant accepts 1.0 (all-in betting, bustable).
        """
        with pytest.raises(ValueError, match="bet_fraction"):
            frozen_referee_evalue(np.array([1.0, 2.0]), bet_fraction=1.0)


# ---------------------------------------------------------------------------
# CandidateSubmission immutability
# ---------------------------------------------------------------------------


class TestCandidateSubmissionImmutability:
    def test_submission_is_frozen_dataclass(self) -> None:
        """Recorded submissions cannot be tampered with (frozen=True).

        The proposer agent must not be able to rewrite history: field
        assignment raises FrozenInstanceError.  A `frozen=False` mutant makes
        the assignments succeed and fails this test.
        """
        sub = CandidateSubmission("c1", submission_time=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sub.candidate_id = "tampered"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            sub.submission_time = 0  # type: ignore[misc]
        assert sub.candidate_id == "c1"
        assert sub.submission_time == 5


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

    def test_alpha_bounds_are_exclusive(self) -> None:
        """alpha must lie in the open interval (0, 1): 0.0, 1.0, 1.5 raise.

        Pins both ends of the guard: `0 <= alpha` and `alpha <= 1` mutants
        accept the closed endpoints, and an `alpha < 2` mutant accepts 1.5.
        """
        with pytest.raises(ValueError, match="alpha"):
            RefereeLedger(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            RefereeLedger(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            RefereeLedger(alpha=1.5)

    def test_default_parameters_pinned_exactly(self) -> None:
        """Pin ledger defaults exactly: null_mean=0, null_std=1, bet_fraction=0.1.

        Exactly one post-submission observation (0.5 at time 1) must give
        e = 1 + 0.1 * 0.5 = 1.05 with n_post_sub_obs = 1.  A drifted default
        null_std (1.0 -> 2.0) yields 1.025 and fails the pin.
        """
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=0))
        verdicts = ledger.evaluate(np.array([0.0, 0.5]))
        assert verdicts[0].e_value == pytest.approx(1.05, abs=1e-12)
        assert verdicts[0].n_post_sub_obs == 1

    def test_no_post_sub_data_verdict_not_admitted(self) -> None:
        """The no-evidence verdict (e=1, n=0) must default to admitted=False.

        A candidate with zero post-submission observations has no evidence and
        must never come back pre-admitted from evaluate().
        """
        ledger = RefereeLedger()
        ledger.submit(CandidateSubmission("c1", submission_time=100))
        verdicts = ledger.evaluate(np.ones(50))
        assert not verdicts[0].admitted


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

    def test_default_alpha_pinned_at_tenth(self) -> None:
        """Pin referee_fdr_control's default alpha=0.1 via threshold math.

        m=2: the k=1 threshold is 2/(1*0.1) = 20 -> e=15 fails; the k=2
        threshold is 2/(2*0.1) = 10 -> e=1 fails; nothing is admitted.  A
        mutant default of 1.1 drops the k=1 threshold to ~1.82 and admits
        everything.
        """
        verdicts = [
            RefereeVerdict("c1", e_value=15.0, n_post_sub_obs=10),
            RefereeVerdict("c2", e_value=1.0, n_post_sub_obs=10),
        ]
        result = referee_fdr_control(verdicts)
        assert not result[0].admitted
        assert not result[1].admitted

    def test_k_one_only_still_admits_top_candidate(self) -> None:
        """The e-BH scan must start at k=1: only the k=1 threshold passes here.

        m=2, alpha=0.1: k=1 threshold 2/(1*0.1) = 20 -> e=30 passes; k=2
        threshold 2/(2*0.1) = 10 -> e=1 fails.  A mutant range starting at
        k=2 skips the only passing step and admits nobody.
        """
        verdicts = [
            RefereeVerdict("c1", e_value=30.0, n_post_sub_obs=10),
            RefereeVerdict("c2", e_value=1.0, n_post_sub_obs=10),
        ]
        result = referee_fdr_control(verdicts, alpha=0.1)
        assert result[0].admitted
        assert not result[1].admitted

    def test_threshold_boundary_is_inclusive(self) -> None:
        """e == m/(k*alpha) exactly must be admitted (>=, not >).

        m=1, alpha=0.1: the threshold is 1/(1*0.1) = 10.0 in exact IEEE
        arithmetic.  e=10.0 sits on the boundary and is admitted; the next
        representable double below it is rejected.  A strict `>` mutant fails
        the boundary half of this pin.
        """
        at_boundary = referee_fdr_control(
            [RefereeVerdict("c1", e_value=10.0, n_post_sub_obs=5)], alpha=0.1
        )
        assert at_boundary[0].admitted
        below = float(np.nextafter(10.0, 0.0))
        below_boundary = referee_fdr_control(
            [RefereeVerdict("c1", e_value=below, n_post_sub_obs=5)], alpha=0.1
        )
        assert not below_boundary[0].admitted


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

    def test_defaults_match_explicit_documented_values(self) -> None:
        """Pin signature defaults: true_edge=0.5, seed=42, alpha=0.1, submission_time=0.

        Calling with defaults must equal calling with the documented values
        explicitly.  Any drifted default changes the SYNTHETIC planted world
        and breaks this equality (seed 43 shifts every draw; edge 1.5 changes
        the planted means; submission_time 1 shifts the evaluation window).
        """
        implicit = leaky_referee_contrast(n_true=5, n_noise=8, n_periods=80)
        explicit = leaky_referee_contrast(
            n_true=5,
            n_noise=8,
            n_periods=80,
            true_edge=0.5,
            seed=42,
            alpha=0.1,
            submission_time=0,
        )
        assert implicit == explicit

    def test_default_world_snapshot_pinned(self) -> None:
        """Hardcoded snapshot of the default SYNTHETIC world (5 true, 8 noise, 80 periods).

        Deterministic (seed=42, edge=0.5, alpha=0.1, submission_time=0);
        regenerate with the pinned call.  This is correctness evidence only,
        never market evidence.  The knife-edge frozen_true_admitted=0.0 pins
        the post-submission slice at `submission_time + 1`: a `+ 2` mutant
        shifts the evaluation window one period and flips it to 4.0; drifted
        defaults also break the snapshot (seed 43 -> 3.0, edge 1.5 -> 5.0,
        submission_time 1 -> 4.0).
        """
        result = leaky_referee_contrast(n_true=5, n_noise=8, n_periods=80)
        assert result == {
            "frozen_true_admitted": 0.0,
            "frozen_noise_admitted": 0.0,
            "leaky_true_admitted": 3.0,
            "leaky_noise_admitted": 0.0,
            "noise_admission_ratio": 0.0,
            "n_true": 5.0,
            "n_noise": 8.0,
            "n_periods": 80.0,
            "alpha": 0.1,
        }

    def test_admission_counts_and_ratio_pinned_exactly(self) -> None:
        """Exact-count pin in a SYNTHETIC world where every admission bucket is non-zero.

        Planted world (seed=21, alpha=0.5, edge=0.4): the frozen referee
        admits 2 true + 1 noise, the leaky referee admits 2 true + 2 noise,
        and noise_admission_ratio = 2 / max(1, 1) = 2.0.  Doubling any of the
        four `sum(1 for ...)` tallies breaks its exact count, and a
        `max(frozen_noise, 2)` ratio-floor mutant reports 1.0 instead of 2.0.
        Correctness evidence only, never market evidence.
        """
        result = leaky_referee_contrast(
            n_true=2,
            n_noise=20,
            n_periods=800,
            true_edge=0.4,
            seed=21,
            alpha=0.5,
            submission_time=0,
        )
        assert result["frozen_true_admitted"] == 2.0
        assert result["frozen_noise_admitted"] == 1.0
        assert result["leaky_true_admitted"] == 2.0
        assert result["leaky_noise_admitted"] == 2.0
        assert result["noise_admission_ratio"] == 2.0


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

    def test_referee_evaluate_uses_ledger_alpha_by_default(self) -> None:
        """alpha=None must fall back to the ledger's frozen alpha.

        One candidate with a single post-submission observation r=50 gives
        e = 1 + 0.1 * 50 = 6.0 exactly; with m=1 the e-BH threshold is
        1/alpha, so the ledger's alpha=0.25 (threshold 4) admits.  Mutants
        that invert or drop the `alpha is None` branch either crash with
        alpha=None (threshold m/(k*None) -> TypeError) or ignore the ledger
        default.
        """
        ledger = RefereeLedger(alpha=0.25)
        ledger.submit(CandidateSubmission("c1", submission_time=0))
        verdicts = referee_evaluate(ledger, np.array([0.0, 50.0]))
        assert verdicts[0].e_value == pytest.approx(6.0, abs=1e-12)
        assert verdicts[0].admitted

    def test_referee_evaluate_explicit_alpha_overrides_ledger(self) -> None:
        """An explicit alpha must take precedence over the ledger's alpha.

        Same world as above (e = 6.0, m = 1): at alpha=0.1 the threshold is
        10 and the candidate must NOT be admitted, even though the ledger
        carries alpha=0.25.  A mutant that always uses ledger.alpha admits.
        """
        ledger = RefereeLedger(alpha=0.25)
        ledger.submit(CandidateSubmission("c1", submission_time=0))
        verdicts = referee_evaluate(ledger, np.array([0.0, 50.0]), alpha=0.1)
        assert verdicts[0].e_value == pytest.approx(6.0, abs=1e-12)
        assert not verdicts[0].admitted
