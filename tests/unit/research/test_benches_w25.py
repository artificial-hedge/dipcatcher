"""Wave-25 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w25
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "tda_persistence",
    "fractional_ou",
    "fourier_hermite",
    "kernel_changepoint",
    "kinetic_ising",
    "heterogeneous_abm",
)


@pytest.fixture(scope="module")
def tda_blob() -> dict[str, float]:
    return benches_w25.bench_tda_persistence()


@pytest.fixture(scope="module")
def fou_blob() -> dict[str, float]:
    return benches_w25.bench_fractional_ou()


@pytest.fixture(scope="module")
def hermite_blob() -> dict[str, float]:
    return benches_w25.bench_fourier_hermite()


@pytest.fixture(scope="module")
def kernel_cp_blob() -> dict[str, float]:
    return benches_w25.bench_kernel_changepoint()


@pytest.fixture(scope="module")
def ising_blob() -> dict[str, float]:
    return benches_w25.bench_kinetic_ising()


@pytest.fixture(scope="module")
def abm_blob() -> dict[str, float]:
    return benches_w25.bench_heterogeneous_abm()


def test_wave25_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w25.bench_tda_persistence, "tda_blob"),
        (benches_w25.bench_fractional_ou, "fou_blob"),
        (benches_w25.bench_fourier_hermite, "hermite_blob"),
        (benches_w25.bench_kernel_changepoint, "kernel_cp_blob"),
        (benches_w25.bench_kinetic_ising, "ising_blob"),
        (benches_w25.bench_heterogeneous_abm, "abm_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_tda_persistence_blob_separates_regimes(tda_blob) -> None:
    assert tda_blob["synthetic_regime_auc"] > 0.8
    assert tda_blob["synthetic_landscape_sep"] > 0.0
    assert tda_blob["synthetic_bottleneck_err"] < 0.5


def test_fractional_ou_blob_recovers_parameters(fou_blob) -> None:
    assert fou_blob["synthetic_h_err_mean"] < 0.25
    assert fou_blob["synthetic_sigma_relerr"] < 0.5
    assert fou_blob["synthetic_autocov_l2_relerr"] < 0.5


def test_fourier_hermite_blob_recovers_moments(hermite_blob) -> None:
    assert hermite_blob["synthetic_call_relerr_mean"] < 0.05
    assert hermite_blob["synthetic_exkurt_err"] < 0.1
    assert hermite_blob["synthetic_neg_mass_o4"] < 0.5


def test_kernel_changepoint_blob_detects_shifts(kernel_cp_blob) -> None:
    assert kernel_cp_blob["synthetic_detect_rate_size0"] < 0.3
    assert kernel_cp_blob["synthetic_detect_rate_size2"] > 0.5
    assert kernel_cp_blob["synthetic_mmd_edge"] > 0.0


def test_kinetic_ising_blob_recovers_couplings(ising_blob) -> None:
    assert ising_blob["synthetic_pl_beats_nmf"] > 0.5
    assert ising_blob["synthetic_pl_J_relerr"] < 0.3
    assert ising_blob["synthetic_enum_paircorr_gap"] < 0.3


def test_heterogeneous_abm_blob_reproduces_switching(abm_blob) -> None:
    assert abm_blob["synthetic_dominance_high_beta"] >= abm_blob["synthetic_dominance_low_beta"]
    assert abm_blob["synthetic_fraction_bounds_ok"] > 0.5
    assert abm_blob["synthetic_determinism"] > 0.5
