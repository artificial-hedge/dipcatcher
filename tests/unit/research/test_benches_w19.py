"""Tests for research/benches_w19.py — SOTA canon wave 19 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the
values are deterministic (seeded from module constants), so re-running a
bench per test buys nothing. The determinism test at the bottom re-runs
each numpy bench once and compares against its fixture bit-for-bit.

The ``ivs_diffusion`` family is TORCH-GATED and ``event_time_flow`` may be
import-absent while its lane branch is in flight — both are handled by the
clean-blob contract (``{}`` is legitimate: wave-12 ``deep_hedging`` /
wave-16 ``rl_market_maker`` precedent), with the forbidden-key scan
running on whatever the bench emits.

Tolerance policy: the wave-19 benches run each module's own shrunk
fixture (the lane suites in tests/unit/{execution,models,metrics} pin the
tight windows); the science assertions below are DIRECTIONAL — they pin
the qualitative claims of the cited papers (Langevin impact's sqrt
mid-regime and vol scaling, Fukasawa's three convergence regimes, RCCP's
lowest-Winkler claim, DCP's CQR reduction) without the lane suites' tight
bands.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w19 import (
    bench_dcp,
    bench_event_time_flow,
    bench_fukasawa_iv,
    bench_ivs_diffusion,
    bench_langevin_impact,
    bench_rccp,
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

_FAMILIES = (
    "langevin_impact",
    "event_time_flow",
    "fukasawa_iv",
    "ivs_diffusion",
    "rccp",
    "dcp",
)
# The numpy/scipy families whose modules have landed on this branch.
_LANDED_NUMPY_BLOBS = (
    "langevin_impact",
    "event_time_flow",
    "fukasawa_iv",
    "rccp",
    "dcp",
)
# May return {} (torch extra absent).
_SOFT_BLOBS = ("ivs_diffusion",)
# Torch-gated family whose science leg runs only when the blob emitted.
_TORCH_BLOBS = ("ivs_diffusion",)


@pytest.fixture(scope="module")
def langevin_impact() -> dict[str, float]:
    return bench_langevin_impact()


@pytest.fixture(scope="module")
def event_time_flow() -> dict[str, float]:
    return bench_event_time_flow()


@pytest.fixture(scope="module")
def fukasawa_iv() -> dict[str, float]:
    return bench_fukasawa_iv()


@pytest.fixture(scope="module")
def ivs_diffusion() -> dict[str, float]:
    return bench_ivs_diffusion()


@pytest.fixture(scope="module")
def rccp() -> dict[str, float]:
    return bench_rccp()


@pytest.fixture(scope="module")
def dcp() -> dict[str, float]:
    return bench_dcp()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


def test_w18_families_registered_as_optional() -> None:
    # The wave-18 registry block landed on w18-canon with the w18 benches;
    # pin it here since the w19 test file owns the wiring check forward.
    for fam in (
        "gslice",
        "neural_sde",
        "stocbench",
        "agentic_lob",
        "fase_eval",
        "kit_paths",
    ):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_LANDED_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


@pytest.mark.parametrize("blob_name", list(_SOFT_BLOBS))
def test_soft_blobs_are_clean(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    # Forbidden-key scan holds whether the bench emitted or returned {}.
    assert family_blob_forbidden_metrics_absent(blob)
    if not blob:
        return
    assert family_blob_has_finite_observation(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_langevin_impact_closed_form_and_sqrt_regime(
    langevin_impact: dict[str, float],
) -> None:
    # Itkin 2026: fresh-pool deterministic impact must match the Eq.-45
    # closed form, and the GLE-pool stochastic impact lands on the
    # paper's Table-3 pin (~0.158) within MC error at this budget.
    blob = langevin_impact
    assert blob["synthetic_fresh_pool_abs_err_t1"] < 1e-4
    assert blob["synthetic_gle_impact_t1"] == pytest.approx(0.158264, abs=0.01)
    assert blob["synthetic_gle_impact_se_t1"] > 0.0
    # The intermediate regime is ~square-root WITHOUT an imposed sqrt law:
    # the local exponent at Q_0 sits near 0.5 between the linear limits.
    assert 0.40 <= blob["synthetic_local_exponent_at_q0"] <= 0.62
    assert blob["synthetic_min_exponent"] >= 0.4
    assert blob["synthetic_fresh_band_width_decades"] > (blob["synthetic_gle_band_width_decades"])


def test_langevin_impact_scaling_and_round_trip(
    langevin_impact: dict[str, float],
) -> None:
    # Vol-proportional thresholds: doubling sigma_x doubles impact in the
    # sqrt regime; horizon-growing thresholds give duration invariance;
    # depletion narrows the sqrt band; round-trip cost is nonnegative.
    blob = langevin_impact
    assert 1.6 <= blob["synthetic_vol_scaling_ratio"] <= 2.4
    assert abs(blob["synthetic_duration_invariance_ratio"] - 1.0) < 0.05
    assert blob["synthetic_depletion_narrows_band"] == 1.0
    assert blob["synthetic_round_trip_min_cost"] >= 0.0
    assert blob["synthetic_latent_displacement_q0"] > 0.0


def test_fukasawa_iv_convergence_regimes(fukasawa_iv: dict[str, float]) -> None:
    # Fukasawa 2026: deterministic-vol reduction is exact; each regime's
    # scaled residual shrinks toward the smaller-order term as the regime
    # parameter is pushed (halving vol-of-vol, n 4->16 mean-reversion,
    # T 0.02->0.08 maturity); the eq. 53 normalization identity holds.
    blob = fukasawa_iv
    assert blob["deterministic_vol_abs_resid"] < 1e-3
    assert blob["small_volvol_abs_resid_a25"] < blob["small_volvol_abs_resid_a50"]
    assert blob["small_volvol_scaled_resid_a25"] < blob["small_volvol_scaled_resid_a50"]
    assert blob["fast_mr_scaled_resid_n16"] < blob["fast_mr_scaled_resid_n4"]
    assert blob["short_mat_scaled_resid_t08"] < blob["short_mat_scaled_resid_t02"]
    assert blob["normalization_gap"] < 0.05
    assert blob["synthetic"] == 1.0


def test_rccp_coverage_and_winkler_ranking(rccp: dict[str, float]) -> None:
    # RCCP paper claims at the shrunk heteroskedastic fixture: the
    # coverage-gap bound is non-vacuous and holds, and retrieval+correction
    # attains the lowest Winkler score of the comparator set with few
    # severe misses; the scalar correction improves on retrieval-only.
    assert rccp["bound_holds"] == 1.0
    assert rccp["bound_vacuous"] == 0.0
    assert rccp["bound"] > 0.0
    assert rccp["coverage"] >= rccp["target_coverage"] - 0.08
    assert rccp["winkler"] <= rccp["scp_winkler"]
    assert rccp["winkler"] <= rccp["recency_winkler"]
    assert rccp["coverage"] > rccp["uncorrected_coverage"]
    assert rccp["severe_miss_rate"] < 0.05
    assert rccp["n_cal_scored"] == 400.0
    assert rccp["n_test"] == 300.0
    assert rccp["synthetic"] == 1.0


def test_dcp_inversion_and_cqr_reduction(dcp: dict[str, float]) -> None:
    # Schweizer et al. 2026: every nonconformity score produces a
    # non-degenerate interval (n_degenerate == 0) meeting the acceptable-
    # coverage floor, the interval-score variant reduces to the CQR
    # baseline (coverage/width near-equal), and the z-score variant beats
    # split-conformal on Winkler at the heteroscedastic oracle.
    for score in ("residual", "zscore", "interval", "knn"):
        assert dcp[f"SYNTHETIC_dcp_{score}_coverage_ok"] == 1.0
        assert dcp[f"SYNTHETIC_dcp_{score}_n_degenerate"] == 0.0
        assert dcp[f"SYNTHETIC_dcp_{score}_mean_width"] > 0.0
    assert dcp["SYNTHETIC_dcp_interval_coverage"] == pytest.approx(
        dcp["SYNTHETIC_cqr_coverage"], abs=0.03
    )
    assert dcp["SYNTHETIC_dcp_interval_mean_width"] == pytest.approx(
        dcp["SYNTHETIC_cqr_mean_width"], abs=0.05
    )
    assert dcp["SYNTHETIC_dcp_zscore_winkler"] < dcp["SYNTHETIC_split_conformal_winkler"]
    assert dcp["SYNTHETIC_n_cal"] == 400.0


def test_event_time_flow_clock_distortions(
    event_time_flow: dict[str, float],
) -> None:
    # Angstmann & Gebbie 2026: the event-time propagator recovers the
    # planted kernel exactly (rel-L2 ~1e-14), the pareto clock produces
    # the mu*gamma calendar-time distortion (sim vs theory slope), and
    # the apparent-kernel MC matches its closed form within a percent.
    blob = event_time_flow
    assert blob["synthetic_kernel_l2_rel_error"] < 1e-8
    assert blob["synthetic_sign_slope_calendar_pareto"] == pytest.approx(
        blob["synthetic_sign_slope_calendar_theory"], abs=0.05
    )
    assert blob["synthetic_impact_slope_calendar_pareto"] == pytest.approx(
        blob["synthetic_impact_slope_calendar_theory"], abs=0.05
    )
    # Operational-time conditioning recovers the true event law in bulk
    # (slope nearer the true -0.5/0.5 than the distorted calendar read).
    assert abs(blob["synthetic_impact_slope_operational"] - 0.5) < abs(
        blob["synthetic_impact_slope_calendar_pareto"] - 0.5
    )
    assert blob["synthetic_apparent_kernel_mc_max_rel_err"] < 0.05
    assert blob["synthetic_event_anchored_max_abs_err"] < 0.1
    assert blob["synthetic_hawkes_fano_max"] > 1.0


def test_ivs_diffusion_noarb_and_hedge(ivs_diffusion: dict[str, float]) -> None:
    if not _HAS_TORCH or not ivs_diffusion:
        pytest.skip("ivs_diffusion requires the nn extra (torch)")
    # Han et al. 2026: the conditional diffusion beats the block-resample
    # generator on energy score, the optimization hedge cuts RMSE and tail
    # error sharply vs resampling, and the no-arb post-training penalty
    # slashes static-arbitrage violation rates below the stream's own.
    blob = ivs_diffusion
    assert blob["es_gain_vs_resample"] > 0.0
    assert blob["es_model"] < blob["es_resample"]
    assert blob["hedge_rmse_model"] < blob["hedge_rmse_resample"]
    assert blob["hedge_es_tail_model"] < blob["hedge_es_tail_resample"]
    assert blob["arb_rate_finetuned"] < blob["arb_rate_generated"]
    assert blob["arb_rate_finetuned"] < blob["arb_rate_data"]
    assert blob["ft_probe_rate_after"] < blob["ft_probe_rate_before"]
    assert blob["n_stream"] > 0.0


def test_numpy_benches_are_deterministic(
    langevin_impact: dict[str, float],
    event_time_flow: dict[str, float],
    fukasawa_iv: dict[str, float],
    rccp: dict[str, float],
    dcp: dict[str, float],
) -> None:
    # Seeded from module constants, so a fresh call must reproduce the
    # fixture bit-for-bit.
    assert bench_langevin_impact() == langevin_impact
    assert bench_event_time_flow() == event_time_flow
    assert bench_fukasawa_iv() == fukasawa_iv
    assert bench_rccp() == rccp
    assert bench_dcp() == dcp
