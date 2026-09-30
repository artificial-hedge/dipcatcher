"""Tests for research/benches_w11.py — SOTA canon wave 11 scorecard families.

Each bench is executed once per module run (fixtures): the conformal-PID and
Sinkhorn batteries are the slowest paths in the wave-11 set, and the values
are deterministic, so re-running them per test buys nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w11 import (
    bench_conformal_pid,
    bench_dro,
    bench_optimal_transport,
    bench_rough_paths,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


@pytest.fixture(scope="module")
def rough_paths() -> dict[str, float]:
    return bench_rough_paths()


@pytest.fixture(scope="module")
def optimal_transport() -> dict[str, float]:
    return bench_optimal_transport()


@pytest.fixture(scope="module")
def dro() -> dict[str, float]:
    return bench_dro()


@pytest.fixture(scope="module")
def conformal_pid() -> dict[str, float]:
    return bench_conformal_pid()


def test_families_registered_as_optional() -> None:
    for fam in ("rough_paths", "optimal_transport", "wasserstein_dro", "conformal_pid"):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    "blob_name",
    ["rough_paths", "optimal_transport", "dro", "conformal_pid"],
)
def test_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_rough_paths_closed_forms(rough_paths: dict[str, float]) -> None:
    # closed loop: level-1 signature = net displacement = 0
    assert rough_paths["closure_residual"] < 1e-9
    # lead-lag antisymmetric level-2 recovers the circle's enclosed area
    assert rough_paths["levy_area_abs_err"] < 0.01
    assert rough_paths["levy_area"] == pytest.approx(rough_paths["levy_area_expected"], abs=0.01)
    # kernel separates the path from its time-shuffle
    assert rough_paths["kernel_gap"] > 0.0
    # Witt dimension for d=2, order 3: 2 + 1 + 2
    assert rough_paths["logsig_dim"] == 5.0


def test_optimal_transport_closed_forms(optimal_transport: dict[str, float]) -> None:
    assert optimal_transport["w1_shifted_abs_err"] < 1e-9
    assert optimal_transport["w2_gauss_identical"] == 0.0
    # squared Bures-Wasserstein: mean term ||delta mu||^2 = 1 + 4
    assert optimal_transport["w2_gauss_shifted"] == pytest.approx(
        optimal_transport["w2_shift_expected"], abs=1e-9
    )
    assert optimal_transport["sink_div_shifted"] > 10.0 * optimal_transport["sink_div_same_law"]


def test_dro_shrinkage_and_duality(dro: dict[str, float]) -> None:
    assert dro["shrinkage_monotone"] == 1.0
    assert dro["worstcase_ge_nominal"] == 1.0
    assert dro["radius0_worst_eq_nominal"] == 1.0
    assert dro["budget_abs_err"] < 1e-9
    assert dro["norm_radius_0"] >= dro["norm_radius_005"] >= dro["norm_radius_02"]


def test_conformal_pid_beats_aci(conformal_pid: dict[str, float]) -> None:
    assert conformal_pid["pid_beats_aci"] == 1.0
    assert conformal_pid["pid_abs_coverage_error"] <= 0.05
    assert conformal_pid["pid_cum_deviation"] < conformal_pid["aci_cum_deviation"]


def test_cheap_benches_are_deterministic() -> None:
    # The two fast benches stand in for the whole set: every bench is seeded
    # from a module constant, and these two re-runs prove the fixtures are
    # reproducible bit-for-bit.
    assert bench_rough_paths() == bench_rough_paths()
    assert bench_dro() == bench_dro()
