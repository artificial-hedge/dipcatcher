"""Wave-26 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w26
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "marchenko_pastur",
    "factor_nowcast",
    "stationary_bootstrap",
    "skill_ratings",
    "modularity_communities",
    "stein_thinning",
)


@pytest.fixture(scope="module")
def mp_blob() -> dict[str, float]:
    return benches_w26.bench_marchenko_pastur()


@pytest.fixture(scope="module")
def dfm_blob() -> dict[str, float]:
    return benches_w26.bench_factor_nowcast()


@pytest.fixture(scope="module")
def sb_blob() -> dict[str, float]:
    return benches_w26.bench_stationary_bootstrap()


@pytest.fixture(scope="module")
def skill_blob() -> dict[str, float]:
    return benches_w26.bench_skill_ratings()


@pytest.fixture(scope="module")
def modularity_blob() -> dict[str, float]:
    return benches_w26.bench_modularity_communities()


@pytest.fixture(scope="module")
def stein_blob() -> dict[str, float]:
    return benches_w26.bench_stein_thinning()


def test_wave26_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w26.bench_marchenko_pastur, "mp_blob"),
        (benches_w26.bench_factor_nowcast, "dfm_blob"),
        (benches_w26.bench_stationary_bootstrap, "sb_blob"),
        (benches_w26.bench_skill_ratings, "skill_blob"),
        (benches_w26.bench_modularity_communities, "modularity_blob"),
        (benches_w26.bench_stein_thinning, "stein_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_marchenko_pastur_blob_recovers_signal(mp_blob) -> None:
    assert mp_blob["synthetic_signal_count_err"] < 0.5
    assert mp_blob["synthetic_mp_bound_viol_rate"] < 0.2
    assert mp_blob["synthetic_clip_fro_improvement"] > 0.0
    assert mp_blob["synthetic_clip_min_eig"] > 0.0


def test_factor_nowcast_blob_recovers_factor(dfm_blob) -> None:
    assert dfm_blob["synthetic_factor_corr"] > 0.8
    assert dfm_blob["synthetic_nowcast_err_ratio"] < 1.0
    assert dfm_blob["synthetic_em_converged"] > 0.5


def test_stationary_bootstrap_blob_calibrates(sb_blob) -> None:
    assert sb_blob["synthetic_coverage_mean"] > 0.6
    assert sb_blob["synthetic_iid_width_ratio"] < 1.0
    assert sb_blob["synthetic_rho_bias"] < 0.25


def test_skill_ratings_blob_recovers_order(skill_blob) -> None:
    assert skill_blob["synthetic_bt_loglik_improve"] > 0.0
    assert skill_blob["synthetic_spearman_recovery"] > 0.7
    assert skill_blob["synthetic_glicko_rd_shrink"] > 0.0


def test_modularity_communities_blob_recovers_partition(modularity_blob) -> None:
    assert modularity_blob["synthetic_ari"] > 0.6
    assert modularity_blob["synthetic_nmi"] > 0.6
    assert modularity_blob["synthetic_k_err"] < 0.5


def test_stein_thinning_blob_compresses(stein_blob) -> None:
    assert stein_blob["synthetic_thin_mmd_edge"] < 1.0
    assert stein_blob["synthetic_herd_edge"] < 1.0
    assert stein_blob["synthetic_ksd_thin_vs_random"] < 1.0
    assert stein_blob["synthetic_determinism"] > 0.5
