"""Wave-29 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w29
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "hmc",
    "proxy_svar",
    "sbi",
    "rqa",
    "hj_distance",
    "tensor_decomp",
)


@pytest.fixture(scope="module")
def hmc_blob() -> dict[str, float]:
    return benches_w29.bench_hmc()


@pytest.fixture(scope="module")
def psvar_blob() -> dict[str, float]:
    return benches_w29.bench_proxy_svar()


@pytest.fixture(scope="module")
def sbi_blob() -> dict[str, float]:
    return benches_w29.bench_sbi()


@pytest.fixture(scope="module")
def rqa_blob() -> dict[str, float]:
    return benches_w29.bench_rqa()


@pytest.fixture(scope="module")
def hj_blob() -> dict[str, float]:
    return benches_w29.bench_hj_distance()


@pytest.fixture(scope="module")
def tensor_blob() -> dict[str, float]:
    return benches_w29.bench_tensor_decomp()


def test_wave29_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w29.bench_hmc, "hmc_blob"),
        (benches_w29.bench_proxy_svar, "psvar_blob"),
        (benches_w29.bench_sbi, "sbi_blob"),
        (benches_w29.bench_rqa, "rqa_blob"),
        (benches_w29.bench_hj_distance, "hj_blob"),
        (benches_w29.bench_tensor_decomp, "tensor_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_hmc_blob_samples(hmc_blob) -> None:
    assert hmc_blob["synthetic_hmc_ess_per_draw"] > 0.5
    assert hmc_blob["synthetic_hmc_accept"] > 0.9
    assert hmc_blob["synthetic_nuts_ess_per_draw"] > 0.05
    assert hmc_blob["synthetic_rhat"] < 1.2
    assert hmc_blob["synthetic_determinism"] == 1.0


def test_proxy_svar_blob_identifies(psvar_blob) -> None:
    assert psvar_blob["synthetic_irf_relerr"] < 0.3
    assert psvar_blob["synthetic_first_stage_f"] > 10.0
    assert psvar_blob["synthetic_corr_z_target"] > 0.5
    assert psvar_blob["synthetic_weak_instrument_flagged"] == 1.0
    assert psvar_blob["synthetic_determinism"] == 1.0


def test_sbi_blob_infers(sbi_blob) -> None:
    assert sbi_blob["synthetic_abc_posterior_err"] < 0.3
    assert sbi_blob["synthetic_smc_posterior_err"] < 0.2
    assert sbi_blob["synthetic_abc_accept_rate"] < 0.1
    assert sbi_blob["synthetic_nre_margin"] > 0.1
    assert sbi_blob["synthetic_determinism"] == 1.0


def test_rqa_blob_orders(rqa_blob) -> None:
    assert rqa_blob["synthetic_det_periodic"] > 0.7
    assert rqa_blob["synthetic_det_iid"] < 0.5
    assert rqa_blob["synthetic_det_lorenz"] > 0.9
    assert rqa_blob["synthetic_det_margin_order"] > 0.5
    assert rqa_blob["synthetic_determinism"] == 1.0


def test_hj_blob_orders(hj_blob) -> None:
    assert hj_blob["synthetic_hj_true"] < 0.2
    assert hj_blob["synthetic_hj_misspec"] > hj_blob["synthetic_hj_true"]
    assert hj_blob["synthetic_krs_p_true"] > 0.05
    assert hj_blob["synthetic_krs_p_misspec"] < 0.05
    assert hj_blob["synthetic_bound_covers_sr"] == 1.0
    assert hj_blob["synthetic_determinism"] == 1.0


def test_tensor_blob_recovers(tensor_blob) -> None:
    assert tensor_blob["synthetic_cp_relerr"] < 0.2
    assert tensor_blob["synthetic_factor_congruence"] > 0.9
    assert tensor_blob["synthetic_core_consistency"] > 80
    assert tensor_blob["synthetic_determinism"] == 1.0
