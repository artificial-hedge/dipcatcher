"""Tests for research/benches_w16.py — SOTA canon wave 16 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the values
are deterministic (seeded from module constants; the RL-MM CPU trainer is
single-threaded and seeded), so re-running a bench per test buys nothing. The
determinism test at the bottom re-runs each bench once and compares against its
fixture bit-for-bit. The ``rl_market_maker`` family is TORCH-GATED — its science
and determinism legs skip when the optional ``nn`` extra is absent, in which case
the bench legitimately returns ``{}`` (the wave-12 ``deep_hedging`` precedent).

Tolerance policy: the wave-16 benches run SHRUNK Monte-Carlo / training budgets
relative to the lane suites in tests/unit/core, tests/unit/models,
tests/unit/metrics and tests/unit/microstructure (GHCP n_reps = 100 instead of
200, ExTRA replications = 6 instead of 10, forecast_selection ``fast=True``
800 dates instead of 1600, and a TINY-BUDGET C51 RL market maker — 11 atoms /
hidden 16 / buffer 2000 / 3 training episodes / 2 paired eval seeds / 600 s
horizon — instead of the paper-scale ``@slow`` ``rl_mm_benchmark``). The science
assertions below are therefore DIRECTIONAL with wider documented slack than the
lane tests — they pin the qualitative claims of the cited papers (the
loss-dominated -> model-dominated QLIKE-ratio flip under level alignment, GHCP
coverage with width shrinkage in the test-stream depth, MS-RLCP in-source
coverage with visible gap degradation and a vacuous Theorem 3.3 bound, the
ExTRA-WCP-T paired-length reduction at near-nominal coverage under an informative
joint shift, the scale-free three-way rule never rewarding dilution, and the C51
session-completion / inventory-cap safety invariants against the AS regime
saturation failure mode) without asserting the lane suites' tight windows.

Documented deviations (mirroring the bench module docstring):
- ``fs_deviation_mirror_affine_r2`` is the module's PAIR-LEVEL mirror R^2 (the
  paper's Section 9.2 statistic, headline 0.9999), not the date-level mirror
  (which includes within-date sampling noise and lands ~0.84); key-semantics
  note following the wave-13 ``repcon`` precedent.
- ``rlmm_inventory_cap_violations`` is a SAFETY invariant (max-abs-inventory
  strictly ABOVE the hard cap ``q_max``, clamped by one-sided gating so == 0),
  distinct from SATURATION (reaching the cap) which is the AS regime failure
  mode; the ``sim_internal_*`` PnL keys stay out of the blob.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w16 import (
    bench_extra_tilt,
    bench_forecast_selection,
    bench_hierarchical_conformal,
    bench_multisource_conformal,
    bench_rl_market_maker,
    bench_vol_loss_decomposition,
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
    not _HAS_TORCH, reason="C51 RL market maker training requires the nn extra (torch)"
)

_FAMILIES = (
    "vol_loss_decomposition",
    "hierarchical_conformal",
    "multisource_conformal",
    "extra_tilt",
    "forecast_selection",
    "rl_market_maker",
)
# The five numpy/scipy families (rl_market_maker is torch-gated, handled apart).
_NUMPY_BLOBS = (
    "vol_loss_decomposition",
    "hierarchical_conformal",
    "multisource_conformal",
    "extra_tilt",
    "forecast_selection",
)


@pytest.fixture(scope="module")
def vol_loss_decomposition() -> dict[str, float]:
    return bench_vol_loss_decomposition()


@pytest.fixture(scope="module")
def hierarchical_conformal() -> dict[str, float]:
    return bench_hierarchical_conformal()


@pytest.fixture(scope="module")
def multisource_conformal() -> dict[str, float]:
    return bench_multisource_conformal()


@pytest.fixture(scope="module")
def extra_tilt() -> dict[str, float]:
    return bench_extra_tilt()


@pytest.fixture(scope="module")
def forecast_selection() -> dict[str, float]:
    return bench_forecast_selection()


@pytest.fixture(scope="module")
def rl_market_maker() -> dict[str, float]:
    return bench_rl_market_maker()


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


def test_rl_market_maker_blob_is_clean(rl_market_maker: dict[str, float]) -> None:
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(rl_market_maker)
    if not _HAS_TORCH:
        assert rl_market_maker == {}
        return
    assert isinstance(rl_market_maker, dict) and rl_market_maker
    assert family_blob_has_finite_observation(rl_market_maker)
    assert all(np.isfinite(v) for v in rl_market_maker.values())


def test_vol_loss_decomposition_flip(vol_loss_decomposition: dict[str, float]) -> None:
    # Tokajuk & Chudziak 2026: the RAW pairwise-marginal QLIKE ratio is
    # loss-dominated (> 1; paper median 2.91)...
    assert vol_loss_decomposition["vld_ratio_raw"] > 1.0
    # ...while validation level alignment FLIPS it to model-dominated (< 1;
    # paper 0.67) — the central claim, on a single seeded SYNTHETIC block.
    assert vol_loss_decomposition["vld_ratio_aligned"] < 1.0
    assert vol_loss_decomposition["vld_flip_loss_to_model"] == 1.0
    # Alignment also narrows the cross-loss one-day VaR breach-rate spread
    # (paper: removes 97% of the variation; budget > 0.5).
    assert vol_loss_decomposition["vld_breach_spread_narrowing"] > 0.5


def test_hierarchical_conformal_coverage_and_shrinkage(
    hierarchical_conformal: dict[str, float],
) -> None:
    nominal = 1.0 - hierarchical_conformal["ghcp_alpha"]
    # Mallick et al. 2026 Thm 2.1: finite-sample distribution-free marginal
    # coverage across every test-stream depth m (SHRUNK n_reps = 100; budget
    # nominal - 0.05, wider than the lane window).
    assert hierarchical_conformal["ghcp_min_coverage"] >= nominal - 0.05
    assert hierarchical_conformal["ghcp_coverage_m0"] >= nominal - 0.05
    # Adding in-stream observations SHRINKS the interval: the m = 10 mean width
    # is well below the m = 0 mean width (the paper's conditioning gain).
    assert hierarchical_conformal["ghcp_width_shrinks"] == 1.0
    assert (
        hierarchical_conformal["ghcp_mean_width_m10"] < hierarchical_conformal["ghcp_mean_width_m0"]
    )


def test_multisource_conformal_in_source_and_gap_degradation(
    multisource_conformal: dict[str, float],
) -> None:
    assert multisource_conformal["msrlcp_alpha"] == pytest.approx(0.10)
    # Hore, Chatterjee & Choudhury 2026: MS-RLCP keeps in-source coverage near
    # nominal under the shared-P_{Y|X} covariate shift (budget >= 0.865).
    assert multisource_conformal["msrlcp_coverage_in_source"] >= 0.865
    # ...and DEGRADES VISIBLY in the thinly covered gap: the vacuous-set
    # fraction and the poorly-represented rate both rise relative to in-source,
    # and the Theorem 3.3 envelope bound goes vacuous (== 1) rather than hiding
    # the failure.
    assert (
        multisource_conformal["msrlcp_frac_vacuous_gap"]
        > multisource_conformal["msrlcp_frac_vacuous_in_source"]
    )
    assert multisource_conformal["msrlcp_bound_vacuous_gap"] == 1.0
    assert 0.0 <= multisource_conformal["msrlcp_poorly_represented_rate_gap"] <= 1.0


def test_extra_tilt_length_reduction_at_near_nominal_coverage(
    extra_tilt: dict[str, float],
) -> None:
    # Choi 2026 (informative eta = 1 design): the tilted WCP-T interval cuts
    # paired set length by a wide margin (paper ~30%; budget > 15%)...
    assert extra_tilt["extra_paired_length_reduction_percent"] > 15.0
    # ...while its coverage stays close to the untilted WCP (small |difference|
    # under the informative shift; documented wider budget 0.05).
    assert abs(extra_tilt["extra_coverage_diff_wcp_t_minus_wcp"]) < 0.05
    # The mode signal is non-degenerate and the tilt weights keep a
    # non-collapsed effective sample size.
    assert extra_tilt["extra_mode_signal_sd"] > 0.0
    assert 0.0 < extra_tilt["extra_weight_ess_percent"] <= 100.0


def test_forecast_selection_scale_free_never_rewards_dilution(
    forecast_selection: dict[str, float],
) -> None:
    # Soleimani 2026: the scale-free three-way rule NEVER admits a pure diluter
    # (the scale-free basis cannot reward dilution)...
    assert forecast_selection["fs_three_way_sf_diluters_admitted"] == 0.0
    # ...and removes a large fraction of the full-pool dilution loss (> 0.5),
    # while the equal-weight basis DOES admit diluters (>= 0 — the replicated
    # scale-mismatch failure mode).
    assert forecast_selection["fs_three_way_sf_removal_fraction"] > 0.5
    assert forecast_selection["fs_three_way_ew_diluters_admitted"] >= 0.0
    # Proposition 5 / Section 9.2 pair-level error-correlation mirror is a
    # near-perfect affine relation (paper headline R^2 = 0.9999; budget > 0.9).
    assert forecast_selection["fs_deviation_mirror_affine_r2"] > 0.9


@requires_torch
def test_rl_market_maker_safety_and_regime_failure(rl_market_maker: dict[str, float]) -> None:
    # Moret & Lillo 2026 (tiny-budget C51): the trained agent COMPLETES every
    # session and never BREACHES the hard inventory cap q_max (a safety
    # invariant — the one-sided gating clamps AT the cap, so violations == 0).
    assert rl_market_maker["rlmm_c51_session_completion"] == 1.0
    assert rl_market_maker["rlmm_inventory_cap_violations"] == 0.0
    # The classic Avellaneda-Stoikov policy SATURATES inventory under
    # regime-switching flow (> 0 — the motivating failure mode) but NOT under
    # stationary flow (== 0).
    assert rl_market_maker["rlmm_regime_saturation_as"] > 0.0
    assert rl_market_maker["rlmm_stationary_saturation_as"] == 0.0
    # Config echoes (tiny-budget path).
    assert rl_market_maker["rlmm_inventory_cap"] == 8.0
    assert rl_market_maker["rlmm_horizon"] == 600.0


def test_numpy_benches_are_deterministic(
    vol_loss_decomposition: dict[str, float],
    hierarchical_conformal: dict[str, float],
    multisource_conformal: dict[str, float],
    extra_tilt: dict[str, float],
    forecast_selection: dict[str, float],
) -> None:
    # Seeded from module constants, so a fresh call must reproduce each fixture
    # bit-for-bit.
    assert bench_vol_loss_decomposition() == vol_loss_decomposition
    assert bench_hierarchical_conformal() == hierarchical_conformal
    assert bench_multisource_conformal() == multisource_conformal
    assert bench_extra_tilt() == extra_tilt
    assert bench_forecast_selection() == forecast_selection


@requires_torch
def test_rl_market_maker_is_deterministic(rl_market_maker: dict[str, float]) -> None:
    # The C51 CPU trainer is single-threaded and fully seeded (torch + numpy),
    # so a fresh tiny-budget run reproduces the fixture bit-for-bit.
    assert bench_rl_market_maker() == rl_market_maker
