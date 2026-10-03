"""Wave 24 bench wiring tests.

Every landed family must emit a finite ``dict[str, float]`` that clears the
catalog honesty gates; soft blobs return ``{}`` while their lane module is
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w24 import (
    bench_breeden_litzenberger,
    bench_fernholz_spt,
    bench_hawkes_em,
    bench_multifractal_vol,
    bench_pmcmc_sv,
    bench_spci_conformal,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_FAMILIES_LANDED = (
    "pmcmc_sv",
    "multifractal_vol",
    "spci_conformal",
    "hawkes_em",
    "fernholz_spt",
    "breeden_litzenberger",
)
_LANDED_NUMPY_BLOBS = _FAMILIES_LANDED


@pytest.fixture(scope="module")
def pmcmc_sv() -> dict[str, float]:
    return bench_pmcmc_sv()


@pytest.fixture(scope="module")
def multifractal_vol() -> dict[str, float]:
    return bench_multifractal_vol()


@pytest.fixture(scope="module")
def spci_conformal() -> dict[str, float]:
    return bench_spci_conformal()


@pytest.fixture(scope="module")
def hawkes_em() -> dict[str, float]:
    return bench_hawkes_em()


@pytest.fixture(scope="module")
def fernholz_spt() -> dict[str, float]:
    return bench_fernholz_spt()


@pytest.fixture(scope="module")
def breeden_litzenberger() -> dict[str, float]:
    return bench_breeden_litzenberger()


def test_landed_families_registered_as_optional() -> None:
    for fam in _FAMILIES_LANDED:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_LANDED_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


# -- science assertions -----------------------------------------------------


def test_pmcmc_sv_posterior_recovery(pmcmc_sv: dict[str, float]) -> None:
    blob = pmcmc_sv
    assert blob["synthetic_post_mu_err"] < 0.1
    assert blob["synthetic_post_phi_err"] < 0.2
    assert blob["synthetic_determinism"] == 1.0


def test_multifractal_scaling(multifractal_vol: dict[str, float]) -> None:
    blob = multifractal_vol
    assert blob["synthetic_lambda2_err"] < 0.05
    assert blob["synthetic_mrw_is_multiscaling"] == 1.0
    assert blob["synthetic_determinism"] == 1.0


def test_spci_coverage_after_shift(spci_conformal: dict[str, float]) -> None:
    blob = spci_conformal
    assert blob["synthetic_spci_coverage_err_post"] < 0.1
    assert blob["synthetic_determinism"] == 1.0


def test_hawkes_em_branching_recovery(hawkes_em: dict[str, float]) -> None:
    blob = hawkes_em
    assert blob["synthetic_branching_fro_err"] < 0.3
    assert blob["synthetic_stability_correct"] == 1.0
    assert blob["synthetic_determinism"] == 1.0


def test_fernholz_master_equation(fernholz_spt: dict[str, float]) -> None:
    blob = fernholz_spt
    assert blob["synthetic_diversity_master_eq_rel_residual"] < 1e-3
    assert blob["synthetic_entropy_master_eq_rel_residual"] < 1e-3
    assert blob["synthetic_local_time_rel_error"] < 0.3


def test_breeden_density_recovery(breeden_litzenberger: dict[str, float]) -> None:
    blob = breeden_litzenberger
    assert blob["synthetic_l1_density_clean"] < 0.2
    assert blob["synthetic_mean_rel_err"] < 0.05
    assert abs(blob["synthetic_mass_defect_after_repair"]) < 0.01
