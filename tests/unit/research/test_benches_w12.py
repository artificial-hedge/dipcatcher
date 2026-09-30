"""Tests for research/benches_w12.py — SOTA canon wave 12 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the
sliced-Wasserstein permutation trials and the torch-gated deep-hedging trainer
are the slowest paths in the wave-12 set, and every value is deterministic
(seeded from a module constant), so re-running them per test buys nothing. The
deep-hedging family is torch-gated — its science test skips when the optional
``nn`` extra is absent, in which case the bench legitimately returns ``{}``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w12 import (
    bench_confidence_sequences,
    bench_deep_hedging,
    bench_mh_enbpi,
    bench_nexcp,
    bench_rwcv,
    bench_score_decomposition,
    bench_sliced_wasserstein,
    bench_stacking,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="deep hedging training requires the nn extra (torch)"
)

_FAMILIES = (
    "nexcp",
    "mh_enbpi",
    "stacking",
    "rwcv",
    "sliced_wasserstein",
    "score_decomposition",
    "confidence_sequences",
    "deep_hedging",
)
# The seven numpy/scipy families (deep_hedging is torch-gated and handled apart).
_NUMPY_BLOBS = (
    "nexcp",
    "mh_enbpi",
    "stacking",
    "rwcv",
    "sliced_wasserstein",
    "score_decomposition",
    "confidence_sequences",
)


@pytest.fixture(scope="module")
def nexcp() -> dict[str, float]:
    return bench_nexcp()


@pytest.fixture(scope="module")
def mh_enbpi() -> dict[str, float]:
    return bench_mh_enbpi()


@pytest.fixture(scope="module")
def stacking() -> dict[str, float]:
    return bench_stacking()


@pytest.fixture(scope="module")
def rwcv() -> dict[str, float]:
    return bench_rwcv()


@pytest.fixture(scope="module")
def sliced_wasserstein() -> dict[str, float]:
    return bench_sliced_wasserstein()


@pytest.fixture(scope="module")
def score_decomposition() -> dict[str, float]:
    return bench_score_decomposition()


@pytest.fixture(scope="module")
def confidence_sequences() -> dict[str, float]:
    return bench_confidence_sequences()


@pytest.fixture(scope="module")
def deep_hedging() -> dict[str, float]:
    return bench_deep_hedging()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_deep_hedging_blob_is_clean(deep_hedging: dict[str, float]) -> None:
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(deep_hedging)
    if not _HAS_TORCH:
        assert deep_hedging == {}
        return
    assert isinstance(deep_hedging, dict) and deep_hedging
    assert family_blob_has_finite_observation(deep_hedging)
    assert all(np.isfinite(v) for v in deep_hedging.values())


def test_nexcp_coverage_in_declared_bracket(nexcp: dict[str, float]) -> None:
    # Empirical coverage under covariate shift lands inside the Theorem 2/3
    # bracket evaluated at the declared TV budget.
    assert nexcp["nexcp_bound_lower"] <= nexcp["nexcp_coverage_shift"]
    assert nexcp["nexcp_coverage_shift"] <= nexcp["nexcp_bound_upper"]
    # Likelihood-ratio weighting restores coverage the uniform band loses.
    assert nexcp["nexcp_coverage_shift"] > nexcp["nexcp_unweighted_coverage"]
    assert nexcp["nexcp_coverage_gap_vs_nominal"] < 0.05
    assert nexcp["nexcp_interval_width_ratio"] > 1.0
    assert 0.0 < nexcp["nexcp_effective_sample_size"] <= 601.0


def test_mh_enbpi_coverage_and_joint_bound(mh_enbpi: dict[str, float]) -> None:
    target = mh_enbpi["mh_enbpi_per_horizon_target"]
    for h in ("h1", "h2", "h3"):
        assert abs(mh_enbpi[f"mh_enbpi_coverage_{h}"] - target) < 0.06
    # Bonferroni joint bound is exactly 1 - alpha under bonferroni_joint.
    assert mh_enbpi["mh_enbpi_joint_coverage_bound"] == pytest.approx(0.9, abs=1e-9)
    # Realized joint coverage sits at/above the bound up to MC slack (EnbPI's
    # per-horizon guarantee is approximate, not finite-sample distribution-free).
    assert mh_enbpi["mh_enbpi_joint_coverage"] >= mh_enbpi["mh_enbpi_joint_coverage_bound"] - 0.05
    # Longer horizons carry wider intervals.
    assert mh_enbpi["mh_enbpi_width_growth_h3_h1"] > 1.0


def test_stacking_recovers_true_generator(stacking: dict[str, float]) -> None:
    assert stacking["stacking_true_weight"] > 0.5
    assert stacking["stacking_true_weight_recovered"] == 1.0
    # Stacking never loses log score / CRPS versus the best single candidate.
    assert stacking["stacking_logscore_gain"] >= -1e-9
    assert stacking["stacking_crps_gain"] >= -1e-9
    # And strictly beats equal weighting (which carries the misspecified models).
    assert stacking["stacking_vs_equal_logscore_gain"] > 0.0
    assert stacking["pseudo_bma_weight_top1"] > 0.5
    assert stacking["pseudo_bma_plus_weight_top1"] > 0.5
    # Recovered weights are sparse (near-zero entropy).
    assert stacking["stacking_weight_entropy"] < 0.1


def test_rwcv_regime_weighting_helps_highvol(rwcv: dict[str, float]) -> None:
    assert rwcv["rwcv_highvol_coverage_gain"] > 0.0
    assert rwcv["rwcv_highvol_coverage"] > rwcv["rwcv_unweighted_highvol_coverage"]
    assert abs(rwcv["rwcv_unconditional_coverage"] - (1.0 - rwcv["rwcv_alpha"])) < 0.06
    assert rwcv["rwcv_mean_width"] > 0.0


def test_sliced_wasserstein_properties(sliced_wasserstein: dict[str, float]) -> None:
    assert sliced_wasserstein["sw_d1_exactness_err"] < 1e-9
    assert sliced_wasserstein["sw_self_distance"] < 1e-9
    assert sliced_wasserstein["sw_null_size"] <= 0.10
    assert sliced_wasserstein["sw_shift_detection_power"] >= 0.8
    assert sliced_wasserstein["sw_barycenter_identical_converged"] == 1.0


def test_score_decomposition_identities(score_decomposition: dict[str, float]) -> None:
    assert abs(score_decomposition["broecker_decomp_error"]) < 1e-9
    assert abs(score_decomposition["kolassa_decomp_error"]) < 1e-9
    assert abs(score_decomposition["brier_identity_error"]) < 1e-9
    assert abs(score_decomposition["sharpness_plus_error_vs_crps_gap"]) < 1e-9
    # Strictly proper RPS: the true pmf beats a perturbed (uniform) pmf.
    assert score_decomposition["rps_at_truth_optimality_gap"] > 0.0


def test_confidence_sequences_coverage_and_width(confidence_sequences: dict[str, float]) -> None:
    assert confidence_sequences["cs_time_uniform_coverage"] >= 0.90
    assert confidence_sequences["cs_violation_rate"] <= 0.10
    assert confidence_sequences["cs_poly_stitching_coverage"] >= 0.90
    assert confidence_sequences["cs_empirical_bernstein_coverage"] >= 0.90
    assert confidence_sequences["cs_wsr_coverage"] >= 0.90
    assert confidence_sequences["cs_width_ratio_vs_fixed"] <= 2.0
    assert -0.65 <= confidence_sequences["cs_width_decay_slope"] <= -0.35


@requires_torch
def test_deep_hedging_reduces_risk(deep_hedging: dict[str, float]) -> None:
    assert deep_hedging["dh_hedged_risk"] < deep_hedging["dh_unhedged_risk"]
    assert deep_hedging["dh_risk_reduction"] > 0.0
    assert np.isfinite(deep_hedging["dh_friction_gap_vs_bs_delta"])


def test_cheap_benches_are_deterministic() -> None:
    # The fast benches stand in for the whole set: every bench is seeded from a
    # module constant, so these re-runs prove the fixtures are reproducible
    # bit-for-bit. The two slow benches (sliced-Wasserstein permutations,
    # deep-hedging torch training) are seeded identically by construction.
    assert bench_nexcp() == bench_nexcp()
    assert bench_mh_enbpi() == bench_mh_enbpi()
    assert bench_stacking() == bench_stacking()
    assert bench_rwcv() == bench_rwcv()
    assert bench_score_decomposition() == bench_score_decomposition()
    assert bench_confidence_sequences() == bench_confidence_sequences()


@requires_torch
def test_deep_hedging_is_deterministic() -> None:
    assert bench_deep_hedging() == bench_deep_hedging()
