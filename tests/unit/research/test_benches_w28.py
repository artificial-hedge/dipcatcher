"""Wave-28 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w28
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "enkf",
    "causal_discovery",
    "knockoffs",
    "callaway_did",
    "surrogate_nonlinear",
    "sindy",
)


@pytest.fixture(scope="module")
def enkf_blob() -> dict[str, float]:
    return benches_w28.bench_enkf()


@pytest.fixture(scope="module")
def cd_blob() -> dict[str, float]:
    return benches_w28.bench_causal_discovery()


@pytest.fixture(scope="module")
def knock_blob() -> dict[str, float]:
    return benches_w28.bench_knockoffs()


@pytest.fixture(scope="module")
def csdid_blob() -> dict[str, float]:
    return benches_w28.bench_callaway_did()


@pytest.fixture(scope="module")
def surr_blob() -> dict[str, float]:
    return benches_w28.bench_surrogate_nonlinear()


@pytest.fixture(scope="module")
def sindy_blob() -> dict[str, float]:
    return benches_w28.bench_sindy()


def test_wave28_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w28.bench_enkf, "enkf_blob"),
        (benches_w28.bench_causal_discovery, "cd_blob"),
        (benches_w28.bench_knockoffs, "knock_blob"),
        (benches_w28.bench_callaway_did, "csdid_blob"),
        (benches_w28.bench_surrogate_nonlinear, "surr_blob"),
        (benches_w28.bench_sindy, "sindy_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_enkf_blob_tracks(enkf_blob) -> None:
    assert enkf_blob["synthetic_enkf_rmse_vs_clim"] < 1.0
    assert enkf_blob["synthetic_eakf_rmse_vs_clim"] < 1.0
    assert enkf_blob["synthetic_letkf_rmse_vs_clim"] <= 1.0
    assert enkf_blob["synthetic_bounded"] == 1.0
    assert enkf_blob["synthetic_determinism"] == 1.0


def test_causal_discovery_blob_recovers(cd_blob) -> None:
    assert cd_blob["synthetic_lingam_dir_acc"] >= 0.6
    assert cd_blob["synthetic_skeleton_f1"] > 0.5
    assert cd_blob["synthetic_vstructure_recall"] > 0.5
    assert cd_blob["synthetic_determinism"] == 1.0


def test_knockoffs_blob_filters(knock_blob) -> None:
    assert knock_blob["synthetic_omp_power"] > 0.7
    assert knock_blob["synthetic_ridge_fdr"] <= 0.25
    assert knock_blob["synthetic_exchange_mean_err"] < 1.0
    assert knock_blob["synthetic_determinism"] == 1.0


def test_callaway_did_blob_identifies(csdid_blob) -> None:
    assert csdid_blob["synthetic_att_bias"] < 0.4
    assert csdid_blob["synthetic_pretrend_p"] > 0.05
    assert csdid_blob["synthetic_pretrend_reject_p"] < 0.05
    assert csdid_blob["synthetic_ci_covers"] >= 0.8
    assert csdid_blob["synthetic_determinism"] == 1.0


def test_surrogate_nonlinear_blob_distinguishes(surr_blob) -> None:
    assert surr_blob["synthetic_bds_power_tent"] > 0.8
    assert surr_blob["synthetic_bds_size_ar"] < 0.4
    assert surr_blob["synthetic_surr_p_tent"] < 0.15
    assert surr_blob["synthetic_surr_p_ar"] > 0.1
    assert surr_blob["synthetic_determinism"] == 1.0


def test_sindy_blob_identifies(sindy_blob) -> None:
    assert sindy_blob["synthetic_coef_relerr"] < 0.05
    assert sindy_blob["synthetic_traj_relerr"] < 0.01
    assert sindy_blob["synthetic_noisy_support_jaccard"] >= 0.8
    assert sindy_blob["synthetic_determinism"] == 1.0
