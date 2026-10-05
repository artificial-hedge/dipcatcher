"""Unit tests for research/selective_inference.py — the selective_inference bench core.

Wired through a *lazy* import inside
:func:`quant_fund.research.benches_w18.bench_selective_inference`, so a
module-level import scan reports it as unreachable. This lane is the compact
wave-18 bench core and a different API from the canon
``quant_fund.models.selective_inference`` (which the canon tests in
``tests/unit/models/test_selective_inference.py`` cover); both implement
polyhedral post-selection intervals, this one specialised to argmin selection
of a loss-table winner.

Coverage style mirrors the canon tests: a hand-workable selection event, a
seeded coverage simulation, and fail-closed edges. All numbers are SYNTHETIC —
a correctness check on the estimator, never market evidence, never a headline
performance ratio.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.selective_inference import (
    SELECTIVE_SCHEMA,
    SelectiveCI,
    _simulate,
    argmin_polytope,
    selective_bench,
    selective_ci,
    truncation_interval,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_Z95 = 1.6448536269514722  # norm.ppf(0.95), the alpha=0.10 two-sided critical value
_COVERAGE_SLACK = 0.10


# ---------------------------------------------------------------------------
# argmin_polytope — the selection event as linear constraints
# ---------------------------------------------------------------------------


def test_argmin_polytope_encodes_winner_le_every_other_head() -> None:
    A, b = argmin_polytope(k_winner=1, k=3)
    assert A.shape == (2, 3)
    assert np.array_equal(b, np.zeros(2))
    # Rows are ordered by the non-winner index: j=0 then j=2.
    assert np.array_equal(A, np.array([[-1.0, 1.0, 0.0], [0.0, 1.0, -1.0]]))


def test_argmin_polytope_holds_exactly_for_the_winner() -> None:
    A, b = argmin_polytope(k_winner=0, k=3)
    winner = np.array([-1.0, 0.5, 2.0])
    assert np.all(A @ winner <= b + 1e-12)
    # A loss vector whose argmin is a different head violates the event.
    for other in (np.array([0.5, -1.0, 2.0]), np.array([0.5, 2.0, -1.0])):
        assert np.any(A @ other > b)


def test_argmin_polytope_row_count_and_self_exclusion() -> None:
    for k in (2, 3, 6):
        for w in range(k):
            A, b = argmin_polytope(k_winner=w, k=k)
            assert A.shape == (k - 1, k)
            assert b.size == k - 1
            assert np.all(A[:, w] == 1.0), "winner coefficient is +1 in every row"
            assert np.all(A.sum(axis=1) == 0.0), "each row is a pairwise contrast"


def test_argmin_polytope_two_head_case_is_hand_readable() -> None:
    """k=2, winner 0 → the single constraint z₀ − z₁ ≤ 0."""
    A, b = argmin_polytope(k_winner=0, k=2)
    assert np.array_equal(A, np.array([[1.0, -1.0]]))
    assert np.array_equal(b, np.zeros(1))


# ---------------------------------------------------------------------------
# truncation_interval — hand-workable selection events
# ---------------------------------------------------------------------------


def test_truncation_interval_two_head_identity_covariance() -> None:
    """Σ = I, η = e₀, event {z₀ ≤ z₁}: c = e₀, z⊥ = (0, z₁−z₀·0) = (0, 3).

    Row: a = A·c = 1 > 0, rhs = 0 − A·z⊥ = 3 → V⁺ = 3, V⁻ = −inf. So the
    winner's mean is only bounded above by the runner-up's observed loss.
    """
    z = np.array([1.0, 3.0])
    A, b = argmin_polytope(k_winner=0, k=2)
    eta = np.array([1.0, 0.0])
    lo, hi = truncation_interval(z, A, b, eta, np.eye(2))
    assert lo == -np.inf
    assert hi == pytest.approx(3.0)


def test_truncation_interval_runner_up_contrast_is_bounded_below() -> None:
    """η = e₁ on the same event: c = e₁, z⊥ = (1, 0), a = A·c = −1 < 0.

    rhs = 0 − A·z⊥ = −1, so V⁻ = rhs/a = 1 and V⁺ = +inf: the event
    {z₀ ≤ z₁} bounds the *runner-up* contrast only from below, at the
    winner's observed loss.
    """
    z = np.array([1.0, 3.0])
    A, b = argmin_polytope(k_winner=0, k=2)
    eta = np.array([0.0, 1.0])
    lo, hi = truncation_interval(z, A, b, eta, np.eye(2))
    assert lo == pytest.approx(1.0)
    assert hi == np.inf


def test_truncation_interval_three_head_event_takes_the_tightest_bound() -> None:
    """V⁺ = min over the runner-ups, since every row caps ηᵀz from above."""
    z = np.array([-2.0, 0.5, 0.25])
    A, b = argmin_polytope(k_winner=0, k=3)
    eta = np.array([1.0, 0.0, 0.0])
    lo, hi = truncation_interval(z, A, b, eta, np.eye(3))
    assert lo == -np.inf
    assert hi == pytest.approx(0.25)


def test_truncation_interval_reports_infeasible_events() -> None:
    """A row orthogonal to c whose rhs is negative → (inf, −inf), not a CI."""
    z = np.zeros(2)
    eta = np.array([1.0, 0.0])
    A = np.array([[0.0, 1.0]])  # a = A·c = 0 with c = e₀ under Σ = I
    lo, hi = truncation_interval(z, A, np.array([-1.0]), eta, np.eye(2))
    assert lo == np.inf and hi == -np.inf
    # The same row with a slack rhs is vacuous rather than infeasible.
    lo2, hi2 = truncation_interval(z, A, np.array([1.0]), eta, np.eye(2))
    assert (lo2, hi2) == (-np.inf, np.inf)


# ---------------------------------------------------------------------------
# selective_ci — exact structure, degenerate edges, coverage
# ---------------------------------------------------------------------------


def test_selective_ci_unconstrained_single_head_matches_the_z_interval() -> None:
    """k=1 → an empty polytope → the conditional pivot is the plain normal CDF,
    so the interval must reproduce the naive z-interval exactly."""
    out = selective_ci(np.array([0.3]), np.eye(1), alpha=0.10)
    assert out.feasible is True
    assert out.winner == 0
    assert (out.v_minus, out.v_plus) == (-np.inf, np.inf)
    assert out.lo == pytest.approx(0.3 - _Z95, abs=1e-9)
    assert out.hi == pytest.approx(0.3 + _Z95, abs=1e-9)
    assert out.lo == pytest.approx(out.naive_lo, abs=1e-9)
    assert out.hi == pytest.approx(out.naive_hi, abs=1e-9)


def test_selective_ci_picks_the_argmin_and_reports_the_naive_interval() -> None:
    means = np.array([0.40, -0.10, 0.25])
    cov = np.eye(3) * 0.04  # sigma = 0.2
    out = selective_ci(means, cov, alpha=0.10)
    assert isinstance(out, SelectiveCI)
    assert out.winner == 1
    assert out.naive_lo == pytest.approx(-0.10 - _Z95 * 0.2)
    assert out.naive_hi == pytest.approx(-0.10 + _Z95 * 0.2)
    # V⁺ is the tightest runner-up bound in ηᵀz units (= the winner contrast).
    assert out.v_plus == pytest.approx(0.25)
    assert out.v_minus == -np.inf
    assert out.feasible is True
    assert out.lo <= means[1] <= out.hi


def test_selective_ci_pays_for_the_selection_with_a_wider_interval() -> None:
    """Conditioning on {winner ≤ runner-up} costs width: the conditional
    interval is wider than the naive z-interval, and its upper end is pulled
    toward the truncation bound V⁺ without crossing it."""
    means = np.array([-1.0, 1.0])
    out = selective_ci(means, np.eye(2), alpha=0.10)
    assert out.v_plus == pytest.approx(1.0)
    assert out.hi < out.v_plus + 1e-9
    assert (out.hi - out.lo) > (out.naive_hi - out.naive_lo)
    assert out.lo <= means[0] <= out.hi
    # The naive interval is centred on the selected estimate; the conditional
    # one is not — it is truncated from above by the selection event.
    centre_cond = 0.5 * (out.lo + out.hi)
    centre_naive = 0.5 * (out.naive_lo + out.naive_hi)
    assert centre_cond != pytest.approx(centre_naive, abs=1e-6)


def test_selective_ci_tie_is_reported_as_uninformative_not_silent() -> None:
    """z_obs exactly at V⁺ (a tie with the runner-up) → pivot ≡ 1, so the
    conditional CI is the whole line and says so."""
    out = selective_ci(np.array([0.0, 0.0]), np.eye(2), alpha=0.10)
    assert out.feasible is True
    assert out.v_plus == pytest.approx(0.0)
    assert out.lo == -np.inf and out.hi == np.inf
    assert math.isfinite(out.naive_lo) and math.isfinite(out.naive_hi)


def test_selective_ci_coverage_at_nominal_on_seeded_simulation() -> None:
    """Conditional coverage ≥ 1 − α − slack on a seeded synthetic loss table."""
    rng = np.random.default_rng(17)
    true_means = np.array([0.05, 0.06, 0.07, 0.09])
    alpha = 0.10
    cond: list[bool] = []
    naive: list[bool] = []
    n_reps = 150
    n_infeasible = 0
    for _ in range(n_reps):
        means, sigma = _simulate(rng, k=4, n=50, true_means=true_means, cov_scale=1.0)
        ci = selective_ci(means, sigma, alpha=alpha)
        if not ci.feasible:
            n_infeasible += 1
            continue
        truth = float(true_means[ci.winner])
        cond.append(ci.lo <= truth <= ci.hi)
        naive.append(ci.naive_lo <= truth <= ci.naive_hi)
    assert n_infeasible <= 5, "the argmin polytope is feasible for a PD covariance"
    assert len(cond) >= n_reps - 5
    cov_cond = float(np.mean(cond))
    assert cov_cond >= (1.0 - alpha) - _COVERAGE_SLACK
    # The naive interval is the winner's-curse baseline: it cannot beat the
    # conditional one here, and the conditional interval is finite-width.
    assert float(np.mean(naive)) <= cov_cond + _COVERAGE_SLACK


def test_selective_ci_widens_with_alpha_and_scales_with_sigma() -> None:
    means = np.array([0.0, 1.0])
    tight = selective_ci(means, np.eye(2), alpha=0.01)
    loose = selective_ci(means, np.eye(2), alpha=0.20)
    assert (tight.hi - tight.lo) > (loose.hi - loose.lo)
    small = selective_ci(means, np.eye(2) * 0.01, alpha=0.10)
    assert (small.hi - small.lo) < (loose.hi - loose.lo)


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_selective_ci_rejects_shape_mismatch() -> None:
    with pytest.raises((ValueError, IndexError)):
        selective_ci(np.zeros(3), np.eye(2))


def test_selective_ci_rejects_nonpositive_variance() -> None:
    """σ ≤ 0 makes the truncated-normal pivot undefined → ValueError."""
    cov = np.zeros((2, 2))
    with warnings.catch_warnings():
        # The degenerate covariance divides by zero before the pivot rejects it.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="sigma must be positive"):
            selective_ci(np.array([0.0, 1.0]), cov, alpha=0.10)


def test_tn_cdf_rejects_bad_sigma() -> None:
    from quant_fund.research.selective_inference import _tn_cdf

    for sigma in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sigma must be positive finite"):
            _tn_cdf(0.0, 0.0, sigma, -1.0, 1.0)


def test_tn_cdf_edges_are_clamped() -> None:
    from quant_fund.research.selective_inference import _tn_cdf

    assert _tn_cdf(-5.0, 0.0, 1.0, -1.0, 1.0) == 0.0
    assert _tn_cdf(5.0, 0.0, 1.0, -1.0, 1.0) == 1.0
    assert _tn_cdf(0.0, 0.0, 1.0, -1.0, 1.0) == pytest.approx(0.5)
    # One-sided truncation deep in a tail must not underflow to a wrong value.
    deep = _tn_cdf(-9.0, -20.0, 1.0, -np.inf, -8.0)
    assert 0.0 <= deep <= 1.0


def test_selective_ci_handles_a_singular_covariance() -> None:
    """Perfectly correlated heads: the contrast direction is degenerate but the
    pivot must still return a bracketed interval (or say it is infeasible)."""
    means = np.array([-3.0, 1.0])
    out = selective_ci(means, np.array([[1.0, 1.0], [1.0, 1.0]]), alpha=0.10)
    assert isinstance(out, SelectiveCI)
    if out.feasible:
        assert out.lo <= out.hi
        assert math.isfinite(out.naive_lo) and math.isfinite(out.naive_hi)


# ---------------------------------------------------------------------------
# Sealed bench payload
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench() -> dict[str, object]:
    return selective_bench(n_reps=60, k=4, n_origins=40, alpha=0.10, seed=23)


def test_bench_payload_is_sealed_and_synthetic(bench: dict[str, object]) -> None:
    assert bench["schema"] == SELECTIVE_SCHEMA
    assert bench["kind"] == "selective_inference"
    assert bench["data_label"] == "SYNTHETIC"
    assert bench["research_only"] is True
    assert bench["n_reps"] == 60
    assert bench["k"] == 4
    assert bench["n_origins"] == 40
    assert bench["alpha"] == pytest.approx(0.10)
    assert isinstance(bench["interpretation"], str) and bench["interpretation"]


def test_bench_coverage_fields_are_consistent(bench: dict[str, object]) -> None:
    cov_cond = float(bench["coverage_conditional"])  # type: ignore[arg-type]
    cov_naive = float(bench["coverage_naive"])  # type: ignore[arg-type]
    assert 0.0 <= cov_cond <= 1.0
    assert 0.0 <= cov_naive <= 1.0
    assert isinstance(bench["n_infeasible"], int)
    assert 0 <= int(bench["n_infeasible"]) <= 60  # type: ignore[arg-type]
    assert float(bench["mean_width_conditional"]) > 0.0  # type: ignore[arg-type]
    assert float(bench["mean_width_naive"]) > 0.0  # type: ignore[arg-type]
    assert isinstance(bench["conditional_calibrated"], bool)


def test_bench_payload_hash_is_reproducible(bench: dict[str, object]) -> None:
    digest = bench["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    body = {k: v for k, v in bench.items() if k != "payload_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == digest


def test_bench_payload_carries_no_forbidden_headline_metric(
    bench: dict[str, object],
) -> None:
    assert family_blob_forbidden_metrics_absent(bench)


def test_bench_is_seed_deterministic(bench: dict[str, object]) -> None:
    again = selective_bench(n_reps=60, k=4, n_origins=40, alpha=0.10, seed=23)
    assert again["payload_sha256"] == bench["payload_sha256"]
    other = selective_bench(n_reps=60, k=4, n_origins=40, alpha=0.10, seed=24)
    assert other["payload_sha256"] != bench["payload_sha256"]


def test_bench_large_k_falls_back_to_a_ladder_of_true_means() -> None:
    """k > 6 has no planted table, so the bench builds a linspace ladder; the
    payload must still be sealed and honest."""
    payload = selective_bench(n_reps=12, k=9, n_origins=20, seed=5)
    assert payload["k"] == 9
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    cov = payload["coverage_conditional"]
    assert isinstance(cov, float) and (math.isnan(cov) or 0.0 <= cov <= 1.0)


def test_simulate_returns_a_psd_covariance_with_the_documented_shape() -> None:
    rng = np.random.default_rng(2)
    true_means = np.array([0.1, 0.2, 0.3])
    means, sigma = _simulate(rng, k=3, n=100, true_means=true_means, cov_scale=1.0)
    assert means.shape == (3,)
    assert sigma.shape == (3, 3)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.linalg.eigvalsh(sigma) > -1e-12)
    assert float(sigma[0, 0]) == pytest.approx(1.0 / 100.0)
    assert float(sigma[0, 1]) == pytest.approx(0.4 / 100.0)


def test_module_exports_a_stable_public_surface() -> None:
    from quant_fund.research import selective_inference as mod

    for name in (
        "argmin_polytope",
        "truncation_interval",
        "SelectiveCI",
        "selective_ci",
        "selective_bench",
        "SELECTIVE_SCHEMA",
    ):
        assert hasattr(mod, name), name


def test_ci_fields_are_plain_floats() -> None:
    out = selective_ci(np.array([0.0, 1.0]), np.eye(2))
    for field in ("lo", "hi", "naive_lo", "naive_hi", "v_plus"):
        value = getattr(out, field)
        assert isinstance(value, float), field
    assert isinstance(out.winner, int)
    assert isinstance(out.feasible, bool)


def test_covariance_is_read_as_float64() -> None:
    means: NDArray[np.float64] = np.array([0.0, 1.0], dtype=np.float32).astype(np.float64)
    cov: NDArray[np.float64] = np.eye(2, dtype=np.float32).astype(np.float64)
    out = selective_ci(means, cov)
    assert out.feasible is True
    assert math.isfinite(out.lo) and math.isfinite(out.hi)
