"""Tests for research/benches_w17.py — SOTA canon wave 17 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the values
are deterministic (seeded from module constants; the stochastic_tracking sweep
is RNG-free), so re-running a bench per test buys nothing. The determinism test
at the bottom re-runs each bench once and compares against its fixture
bit-for-bit. The ``diffpts``, ``multilevel_mm`` and ``rlmm_c51`` families are
TORCH-GATED — their science and determinism legs skip when the optional ``nn``
extra is absent, in which case the bench legitimately returns ``{}`` (the
wave-12 ``deep_hedging`` / wave-16 ``rl_market_maker`` precedent).

Tolerance policy: the wave-17 benches run SHRUNK Monte-Carlo / training budgets
relative to the lane suites in tests/unit/models, tests/unit/metrics,
tests/unit/microstructure and tests/unit/execution (DiffPTS 220/60 train/test +
hidden (16,) + 50 epochs + 20 diffusion steps instead of the module defaults
700/250, (32,32), 150, 50; Cheridito-Weiss hidden (16,) nets and 3 episodes at
horizon 30 instead of the lane's 4 at 40-60; the Algorithm-C bandit loop at the
lane's own tiny budget — pool 6 x 30 expected MOs, 4 episodes at horizon 120;
the SGA DAG certificate at 2000 panels instead of 4000). The science assertions
below are therefore DIRECTIONAL with wider documented slack than the lane tests
— they pin the qualitative claims of the cited papers (weighted-beats-
unweighted conformal coverage under the planted covariate shift, the harmonic
power-shift gap, NFE-budgeted CRPS reporting, completion / inventory-cap safety
invariants for both RL market makers, the variance-bound certificate and
horizon monotonicity of the error DAG, the passive-execution fill / monotone
fluid path / zero-impact exact reduction, and the sqrt(eps) sharp rate of the
regularized-to-unregularized tracking gap) without asserting the lane suites'
tight windows.

Documented deviations (mirroring the bench module docstring):
- ``pim_passive_minus_aggressive`` re-keys the module's
  ``sim_internal_passive_minus_aggressive_mean`` — the forbidden ``pnl`` token
  is dropped at the adapter, and no sign is asserted (the baseline gap is a
  recorded diagnostic, not a claimed win).
- ``st_loglog_slope_cht`` is asserted in a widened [0.35, 0.65] band: the lane
  slope test used a logspace(-6, -2) sweep on a 200-step grid while the bench
  shares the explicit/tracking geomspace((5h)^2, 0.04, 8) sweep on a 400-step
  grid.
- ``rlmm_c51_eval_completion_min`` is the min over the ``c51_*`` combos only
  (the classic AS/GLFT arms are allowed to fail under regime scenarios — that
  saturation failure is the paper's motivating contrast, not a defect).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w17 import (
    bench_diffpts,
    bench_extra_conformal,
    bench_multilevel_mm,
    bench_passive_impact,
    bench_rlmm_c51,
    bench_sga_uq,
    bench_stochastic_tracking,
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
    not _HAS_TORCH, reason="wave-17 torch-gated benches require the nn extra (torch)"
)

_FAMILIES = (
    "diffpts",
    "extra_conformal",
    "multilevel_mm",
    "rlmm_c51",
    "sga_uq",
    "passive_impact",
    "stochastic_tracking",
)
# The four numpy/scipy families (diffpts / multilevel_mm / rlmm_c51 are
# torch-gated and handled apart).
_NUMPY_BLOBS = (
    "extra_conformal",
    "sga_uq",
    "passive_impact",
    "stochastic_tracking",
)
_TORCH_BLOBS = (
    "diffpts",
    "multilevel_mm",
    "rlmm_c51",
)


@pytest.fixture(scope="module")
def diffpts() -> dict[str, float]:
    return bench_diffpts()


@pytest.fixture(scope="module")
def extra_conformal() -> dict[str, float]:
    return bench_extra_conformal()


@pytest.fixture(scope="module")
def multilevel_mm() -> dict[str, float]:
    return bench_multilevel_mm()


@pytest.fixture(scope="module")
def rlmm_c51() -> dict[str, float]:
    return bench_rlmm_c51()


@pytest.fixture(scope="module")
def sga_uq() -> dict[str, float]:
    return bench_sga_uq()


@pytest.fixture(scope="module")
def passive_impact() -> dict[str, float]:
    return bench_passive_impact()


@pytest.fixture(scope="module")
def stochastic_tracking() -> dict[str, float]:
    return bench_stochastic_tracking()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


def test_w16_families_registered_as_optional() -> None:
    # The wave-16 registry block landed on main with the w16 benches; these
    # assertions pin it here since the w17 test file owns the w16 wiring check.
    for fam in ("vol_loss_decomposition", "hierarchical_conformal", "multisource_conformal"):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


@pytest.mark.parametrize("blob_name", list(_TORCH_BLOBS))
def test_torch_blobs_are_clean(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(blob)
    if not _HAS_TORCH:
        assert blob == {}
        return
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_extra_conformal_weighted_beats_unweighted(
    extra_conformal: dict[str, float],
) -> None:
    # Weighted conformal restores coverage under the planted covariate shift;
    # unweighted (plain split) undercovers. Budget keeps the lane floor 0.85.
    assert extra_conformal["xc_coverage"] > extra_conformal["xc_unweighted_coverage"]
    assert extra_conformal["xc_coverage"] >= 0.85
    assert extra_conformal["xc_coverage_error"] < extra_conformal["xc_unweighted_coverage_error"]
    # The harmonic-mean label extension (extra-harm) under the misspecified
    # label tilt: visible power-shift gap and a non-collapsed ESS.
    assert extra_conformal["xc_harm_coverage"] < extra_conformal["xc_harm_unweighted_coverage"]
    assert extra_conformal["xc_harm_coverage_gap"] > 0.1
    assert extra_conformal["xc_harm_ess_fraction"] > 0.2
    assert 0.0 < extra_conformal["xc_ess_fraction"] <= 1.0


def test_sga_uq_certificate_and_coverage(sga_uq: dict[str, float]) -> None:
    # Propagated DAG covariance matches the closed-form VAR(1) covariance
    # (differentiable-programming path vs analytic).
    assert sga_uq["sga_cov_closed_form_max_err"] < 1e-8
    # The variance-bound certificate certifies; variance is horizon-monotone.
    assert sga_uq["sga_certified"] == 1.0
    assert sga_uq["sga_variance_monotone"] == 1.0
    assert sga_uq["sga_terminal_variance"] > 0.0
    # 1.96-sigma intervals cover the seeded error panels near 0.95 nominal
    # (2000-panel MC; budget band [0.85, 1.0]).
    assert 0.85 <= sga_uq["sga_coverage_mean"] <= 1.0
    # uq_from_forecaster: 2 slice levels x 64 branches + root, one node merged
    # away by greedy alignment at the pinned seed.
    assert sga_uq["sga_n_nodes"] == 128.0
    assert sga_uq["sga_graph_complexity"] > 0.0


def test_passive_impact_fill_monotone_and_reduction(
    passive_impact: dict[str, float],
) -> None:
    # The optimal schedule fills a non-trivial share in finite trading time.
    assert 0.0 < passive_impact["pim_fill_fraction_mean"] <= 1.0
    assert passive_impact["pim_trading_time_mean"] > 0.0
    # The fluid inventory relaxation is non-increasing (it only liquidates).
    assert passive_impact["pim_inventory_monotone"] == 1.0
    # Zero-impact/zero-OFI spec reduces exactly: terminal price == mid.
    assert passive_impact["pim_zero_impact_terminal_abs_dev"] < 1e-9
    # m == k selects the closed-form branch (Theorem 3.1).
    assert passive_impact["pim_method_closed_form"] == 1.0


def test_stochastic_tracking_sharp_rate(stochastic_tracking: dict[str, float]) -> None:
    # Theorem 4.2: the unregularized optimum's value matches the closed-form
    # benchmark on the same grid.
    assert stochastic_tracking["st_ow_value_abs_gap"] < 1e-6
    # Prop. 5.9 / Thm 5.7: the regularized-to-unregularized gap converges at
    # the sharp sqrt(eps) rate (log-log slope ~ 0.5).
    assert 0.35 <= stochastic_tracking["st_loglog_slope_cht"] <= 0.65
    assert 0.4 <= stochastic_tracking["st_loglog_slope_explicit"] <= 0.6
    assert 0.4 <= stochastic_tracking["st_loglog_slope_tracking"] <= 0.65
    # c* = 2 sqrt(beta lam) Xi^2 / (beta T + 2)^2 = 2/9 at the pinned params.
    assert stochastic_tracking["st_sharp_rate_constant"] == pytest.approx(2.0 / 9.0)


@requires_torch
def test_diffpts_nfe_budgets_and_scores(diffpts: dict[str, float]) -> None:
    # NFE accounting: DDIM budgets (2, 8) cost their step counts; the DDPM
    # ancestral sampler costs the full 20-step schedule.
    assert diffpts["diffpts_ddim2_nfe"] == 2.0
    assert diffpts["diffpts_ddim8_nfe"] == 8.0
    assert diffpts["diffpts_ddpm_nfe"] == 20.0
    assert diffpts["diffpts_ddim2_nfe"] < diffpts["diffpts_ddim8_nfe"] < diffpts["diffpts_ddpm_nfe"]
    # Proper-score ranges on the shrunk AR(1)-bimodal stream.
    assert diffpts["diffpts_ddpm_crps"] > 0.0
    assert diffpts["diffpts_ddim2_crps"] > 0.0
    assert 0.0 <= diffpts["diffpts_coverage_90"] <= 1.0
    assert 0.0 <= diffpts["diffpts_pit_ks"] <= 1.0
    assert diffpts["diffpts_width_90"] > 0.0
    # StocBench-style NFE trade-off recorded: DDIM-8 stays within 100% of the
    # full DDPM CRPS (the lane's recorded-gap window).
    assert (
        abs(diffpts["diffpts_ddim8_crps"] - diffpts["diffpts_ddpm_crps"])
        <= diffpts["diffpts_ddpm_crps"]
    )


@requires_torch
def test_multilevel_mm_completion_and_cap(multilevel_mm: dict[str, float]) -> None:
    # Cheridito & Weiss 2026 (tiny-budget): the trained level-grouped agent
    # completes every paired-seed session and NO arm breaches the hard
    # inventory cap (a safety invariant — the env clamps at the cap).
    assert multilevel_mm["mlmm_agent_session_completion"] == 1.0
    assert multilevel_mm["mlmm_inventory_cap_violations"] == 0.0
    assert multilevel_mm["mlmm_inventory_cap"] == 8.0
    assert 0.0 <= multilevel_mm["mlmm_glft_session_completion"] <= 1.0
    assert 0.0 <= multilevel_mm["mlmm_random_session_completion"] <= 1.0
    assert multilevel_mm["mlmm_agent_fills_mean"] >= 0.0
    # Spec echoes: paper Table-1 three levels x four lots.
    assert multilevel_mm["mlmm_n_levels"] == 3.0
    assert multilevel_mm["mlmm_lots"] == 4.0


@requires_torch
def test_rlmm_c51_bandit_and_robustness(rlmm_c51: dict[str, float]) -> None:
    # Moret & Lillo Algorithm C (tiny budget): all four bandit-selected
    # fine-tune episodes run their full horizon, and the capped softmax keeps
    # every arm weight at or under w_max = 0.4.
    assert rlmm_c51["rlmm_c51_episodes_completed"] == rlmm_c51["rlmm_c51_episodes"] == 4.0
    assert rlmm_c51["rlmm_c51_bandit_weight_max"] <= 0.4 + 1e-9
    assert rlmm_c51["rlmm_c51_pool_size"] == 6.0
    assert rlmm_c51["rlmm_c51_difficulty_spread"] >= 0.0
    # The trained agent completes every scenario-family eval session and no
    # actor breaches the hard inventory cap.
    assert rlmm_c51["rlmm_c51_eval_completion_min"] == 1.0
    assert rlmm_c51["rlmm_c51_cap_violations"] == 0.0
    assert rlmm_c51["rlmm_c51_inventory_cap"] == 8.0
    assert rlmm_c51["rlmm_c51_horizon"] == 120.0


def test_numpy_benches_are_deterministic(
    extra_conformal: dict[str, float],
    sga_uq: dict[str, float],
    passive_impact: dict[str, float],
    stochastic_tracking: dict[str, float],
) -> None:
    # Seeded from module constants (stochastic_tracking is RNG-free), so a
    # fresh call must reproduce each fixture bit-for-bit.
    assert bench_extra_conformal() == extra_conformal
    assert bench_sga_uq() == sga_uq
    assert bench_passive_impact() == passive_impact
    assert bench_stochastic_tracking() == stochastic_tracking


@requires_torch
def test_torch_benches_are_deterministic(
    diffpts: dict[str, float],
    multilevel_mm: dict[str, float],
    rlmm_c51: dict[str, float],
) -> None:
    # The torch benches seed numpy + torch throughout (single-threaded CPU
    # trainers), so a fresh tiny-budget run reproduces each fixture bit-for-bit.
    assert bench_diffpts() == diffpts
    assert bench_multilevel_mm() == multilevel_mm
    assert bench_rlmm_c51() == rlmm_c51
