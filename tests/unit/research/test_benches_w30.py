"""Wave-30 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w30
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "lp_iv",
    "bispectrum",
    "functional_linear",
    "gas_score",
    "count_data",
    "lyapunov",
)


@pytest.fixture(scope="module")
def lpiv_blob() -> dict[str, float]:
    return benches_w30.bench_lp_iv()


@pytest.fixture(scope="module")
def bispec_blob() -> dict[str, float]:
    return benches_w30.bench_bispectrum()


@pytest.fixture(scope="module")
def flm_blob() -> dict[str, float]:
    return benches_w30.bench_functional_linear()


@pytest.fixture(scope="module")
def gas_blob() -> dict[str, float]:
    return benches_w30.bench_gas_score()


@pytest.fixture(scope="module")
def count_blob() -> dict[str, float]:
    return benches_w30.bench_count_data()


@pytest.fixture(scope="module")
def lyap_blob() -> dict[str, float]:
    return benches_w30.bench_lyapunov()


def test_wave30_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w30.bench_lp_iv, "lpiv_blob"),
        (benches_w30.bench_bispectrum, "bispec_blob"),
        (benches_w30.bench_functional_linear, "flm_blob"),
        (benches_w30.bench_gas_score, "gas_blob"),
        (benches_w30.bench_count_data, "count_blob"),
        (benches_w30.bench_lyapunov, "lyap_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_lpiv_blob_identifies(lpiv_blob) -> None:
    assert lpiv_blob["synthetic_irf_relerr"] < 0.3
    assert lpiv_blob["synthetic_first_stage_f_min"] > 10.0
    assert lpiv_blob["synthetic_lpiv_beats_naive"] == 1.0
    assert lpiv_blob["synthetic_ar_widens_under_weak_iv"] == 1.0
    assert lpiv_blob["synthetic_determinism"] == 1.0


def test_bispectrum_blob_detects(bispec_blob) -> None:
    assert bispec_blob["synthetic_hinich_p_gauss"] > 0.02
    assert bispec_blob["synthetic_hinich_p_quad"] < 0.01
    assert bispec_blob["synthetic_mean_bicoh_quad"] > bispec_blob["synthetic_mean_bicoh_gauss"]
    assert bispec_blob["synthetic_determinism"] == 1.0


def test_functional_linear_blob_recovers(flm_blob) -> None:
    assert flm_blob["synthetic_fpca_relerr"] < 0.3
    assert flm_blob["synthetic_eig1_congruence"] > 0.9
    assert flm_blob["synthetic_flm_r2"] > 0.8
    assert flm_blob["synthetic_beta_relerr"] < 0.1
    assert flm_blob["synthetic_determinism"] == 1.0


def test_gas_score_blob_tracks(gas_blob) -> None:
    assert gas_blob["synthetic_sig2_corr"] > 0.5
    assert gas_blob["synthetic_lam_corr"] > 0.3
    assert gas_blob["synthetic_lr_vs_static"] > 0.0
    assert gas_blob["synthetic_determinism"] == 1.0


def test_count_data_blob_recovers(count_blob) -> None:
    assert count_blob["synthetic_nb_beta_err"] < 0.15
    assert count_blob["synthetic_alpha_err"] < 0.3
    assert count_blob["synthetic_lr_alpha_p"] < 0.01
    assert count_blob["synthetic_pearson_overdisp"] > 1.5
    assert count_blob["synthetic_vuong_picks_zip"] == 1.0
    assert count_blob["synthetic_determinism"] == 1.0


def test_lyapunov_blob_orders(lyap_blob) -> None:
    assert lyap_blob["synthetic_lyap_logistic_err"] < 0.15
    assert lyap_blob["synthetic_lyap_henon_err"] < 0.15
    assert lyap_blob["synthetic_lyap_periodic"] < 0.05
    assert lyap_blob["synthetic_lyap_chaos_ordering"] == 1.0
    assert lyap_blob["synthetic_fnn_dim1_frac"] > lyap_blob["synthetic_fnn_dim2_frac"]
    assert lyap_blob["synthetic_determinism"] == 1.0
