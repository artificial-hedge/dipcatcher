"""Tests for microstructure.agentic_lob — SYNTHETIC correctness only.

Seeded Monte-Carlo checks of the Rosenzweig (2026, arXiv:2609.31260)
phase-transition machinery on the ZI-LOB: three-phase classification
(collapsed / continuous_liquidity / frozen), the (lam, theta) phase
diagram and its boundary estimators, changepoint-coalesced intra-path
phase labels, the sequential anytime-valid phase alarm, the Sec.-4
ensemble impact z-score, initial-condition hysteresis, deterministic
phase clustering, and the flat bench bundle. Every observable is a
correctness diagnostic on a zero-intelligence simulator — never market
evidence — and no headline Sharpe/Sortino/Calmar/NAV key may appear in
any report blob.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from quant_fund.microstructure.agentic_lob import (
    AGENTIC_LOB_REVISION,
    IMPACT_REGIMES,
    PHASE_LABELS,
    PhaseScanConfig,
    bench_agentic_lob,
    classify_impact_regime,
    classify_phase,
    cluster_injection_phases,
    detect_phase_boundaries,
    ensemble_phase,
    hysteresis_sweep,
    impact_zscore,
    impact_zscore_experiment,
    phase_alarm,
    phase_diagram,
    phase_feature_series,
    run_phase_path,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    santa_fe_config,
)

FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

# Small shared scan: ~0.1 s per ensemble at these sizes.
SC = PhaseScanConfig(horizon=160.0, warmup=100.0, n_paths=4, seed=7)


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


def _assert_no_forbidden_keys(blob: Any) -> None:
    low = [k.lower() for k in _all_keys(blob)]
    for tok in FORBIDDEN_HEADLINE_TOKENS:
        assert not any(tok in k for k in low), f"forbidden token {tok!r} in keys"
    for k in low:
        if "pnl" in k:
            assert k == "live_pnl_claim" or k.startswith("sim_internal_"), (
                f"pnl metric key not simulator-internal: {k}"
            )


def _blob_equal(a: Any, b: Any) -> bool:
    """Deep equality where nan == nan (determinism check)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_blob_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_blob_equal(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
    return a == b


# ---------------------------------------------------------------------------
# Module fixtures: seeded ensembles at three control-parameter points
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def starved_cell() -> dict[str, Any]:
    """lam=0.005 starves the strip: collapse rate ~1 (paper's vapor phase)."""
    return ensemble_phase(ZILobConfig(lam=0.005, theta_cxl=0.02, band=5, seed=7), SC)


@pytest.fixture(scope="module")
def santafe_cell() -> dict[str, Any]:
    """Santa-Fe calibration point: continuous-liquidity phase."""
    return ensemble_phase(santa_fe_config(seed=7), SC)


@pytest.fixture(scope="module")
def thick_cell() -> dict[str, Any]:
    """Deep supply (lam=0.5, theta=0.005): frozen phase, sigma ~ 0."""
    return ensemble_phase(ZILobConfig(lam=0.5, theta_cxl=0.005, band=5, seed=7), SC)


# ---------------------------------------------------------------------------
# PhaseScanConfig / classify_phase — closed-form and fail-closed
# ---------------------------------------------------------------------------


def test_config_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError):
        PhaseScanConfig(horizon=0.0)


def test_config_rejects_bad_collapse_threshold() -> None:
    with pytest.raises(ValueError):
        PhaseScanConfig(collapse_rate_threshold=1.5)


def test_config_rejects_bool_n_paths() -> None:
    with pytest.raises(ValueError):
        PhaseScanConfig(n_paths=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sigma", "cr", "expected"),
    [
        (0.30, 0.10, "continuous_liquidity"),
        (0.30, 0.90, "collapsed"),
        (0.01, 0.10, "frozen"),
        (0.05, 0.0, "continuous_liquidity"),  # boundary is strict <
    ],
)
def test_classify_phase_table(sigma: float, cr: float, expected: str) -> None:
    assert classify_phase(sigma, cr, SC) == expected


def test_classify_phase_nan_sigma_with_collapse_is_collapsed() -> None:
    assert classify_phase(float("nan"), 0.7, SC) == "collapsed"


def test_classify_phase_nan_sigma_no_collapse_raises() -> None:
    with pytest.raises(ValueError):
        classify_phase(float("nan"), 0.0, SC)


def test_classify_phase_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        classify_phase(0.2, 2.0, SC)


def test_phase_labels_cover_three_regimes() -> None:
    assert set(PHASE_LABELS) == {"collapsed", "continuous_liquidity", "frozen"}


# ---------------------------------------------------------------------------
# run_phase_path / ensemble_phase — seeded synthetic observables
# ---------------------------------------------------------------------------


def test_starved_cell_is_collapsed(starved_cell: dict[str, Any]) -> None:
    assert starved_cell["phase"] == "collapsed"
    assert starved_cell["collapse_rate"] >= 0.75
    assert starved_cell["mid_defined_frac"] < 0.3
    _assert_no_forbidden_keys(starved_cell)


def test_santafe_cell_is_continuous(santafe_cell: dict[str, Any]) -> None:
    assert santafe_cell["phase"] == "continuous_liquidity"
    assert math.isfinite(santafe_cell["sigma_step_median_ticks"])
    assert santafe_cell["sigma_step_median_ticks"] > 0.1
    assert santafe_cell["mid_defined_frac"] > 0.9


def test_thick_cell_is_frozen(thick_cell: dict[str, Any]) -> None:
    assert thick_cell["phase"] == "frozen"
    assert thick_cell["sigma_step_median_ticks"] < SC.frozen_sigma_step
    assert thick_cell["collapse_rate"] == 0.0
    assert thick_cell["mean_total_depth"] > 100.0


def test_collapse_rate_decreases_in_lam() -> None:
    """Monotone order parameter: more liquidity provision, less collapse."""
    rates = []
    for lam in (0.005, 0.06, 0.3):
        cell = ensemble_phase(ZILobConfig(lam=lam, theta_cxl=0.02, band=5, seed=11), SC)
        rates.append(cell["collapse_rate"])
    assert rates[0] >= rates[1] >= rates[2]
    assert rates[0] - rates[2] >= 0.5


def test_run_phase_path_starved_episode_bookkeeping() -> None:
    p = run_phase_path(ZILobConfig(lam=0.005, theta_cxl=0.02, band=5, seed=3), SC)
    assert p["collapsed"] is True
    assert p["max_empty_seconds"] >= SC.collapse_min_seconds
    assert p["n_empty_episodes"] >= 1
    assert p["empty_time_frac"] > 0.5
    assert p["collapse_time"] is not None
    assert 0.0 < p["collapse_time"] < SC.warmup + SC.horizon


def test_run_phase_path_santafe_resilience_finite() -> None:
    p = run_phase_path(santa_fe_config(seed=3), SC)
    if p["n_empty_episodes"] - p["n_unrecovered_episodes"] >= 1:
        assert p["resilience_mean_seconds"] > 0.0
        assert p["resilience_max_seconds"] >= p["resilience_mean_seconds"]
    assert p["mid_defined_frac"] > 0.9
    # Composed book_phase_metrics blob rides along (single source of truth).
    assert "book_phase" in p and "phase" in p["book_phase"]


def test_run_phase_path_deterministic() -> None:
    cfg = santa_fe_config(seed=42)
    a = run_phase_path(cfg, SC)
    b = run_phase_path(cfg, SC)
    assert _blob_equal(a, b)


def test_run_phase_path_rejects_non_config() -> None:
    with pytest.raises(TypeError):
        run_phase_path({"lam": 0.06}, SC)  # type: ignore[arg-type]


def test_ensemble_phase_n_paths_override() -> None:
    cell = ensemble_phase(santa_fe_config(seed=5), SC, n_paths=2)
    assert cell["n_paths"] == 2
    assert len(cell["sigma_step_per_path"]) == 2


# ---------------------------------------------------------------------------
# phase_diagram — grid shapes and boundary structure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_diagram() -> dict[str, Any]:
    return phase_diagram(
        base_config=ZILobConfig(band=5, seed=7),
        lam_values=(0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4),
        theta_values=(0.005, 0.02, 0.06, 0.12),
        scan=SC,
    )


def test_diagram_grid_shapes(small_diagram: dict[str, Any]) -> None:
    pb = small_diagram["per_band"][0]
    assert len(pb["cells"]) == 4
    assert all(len(row) == 7 for row in pb["cells"])
    assert len(pb["collapse_grid"]) == 4 and len(pb["collapse_grid"][0]) == 7
    assert len(pb["phase_grid"]) == 4 and len(pb["phase_grid"][0]) == 7
    assert small_diagram["n_cells_total"] == 28
    _assert_no_forbidden_keys(small_diagram)


def test_diagram_collapse_frontier_is_monotone(small_diagram: dict[str, Any]) -> None:
    """lam_c(theta) is non-decreasing: heavier cancel pressure needs more
    provision to stay out of the collapsed phase (linear boundary analog)."""
    lam_c = np.asarray(small_diagram["per_band"][0]["lam_c_by_theta"], dtype=np.float64)
    assert np.isfinite(lam_c).all()
    assert np.all(np.diff(lam_c) >= 0.0)
    assert lam_c[0] < lam_c[-1]


def test_diagram_phase_labels_valid(small_diagram: dict[str, Any]) -> None:
    for row in small_diagram["per_band"][0]["phase_grid"]:
        for lab in row:
            assert lab in PHASE_LABELS
    flat = [p for row in small_diagram["per_band"][0]["phase_grid"] for p in row]
    assert flat.count("collapsed") >= 4
    assert flat.count("frozen") >= 1
    assert flat.count("continuous_liquidity") >= 4


def test_diagram_sigma_decreases_with_lam(small_diagram: dict[str, Any]) -> None:
    """Median sigma falls as the book deepens along lam at fixed theta —
    the continuous->frozen direction of the order parameter."""
    sigma = np.asarray(small_diagram["per_band"][0]["sigma_step_grid"], dtype=np.float64)
    col = sigma[0]  # theta = 0.005
    fin = np.isfinite(col)
    assert fin.sum() >= 3
    assert col[fin][-1] <= col[fin][0]


def test_diagram_replicates_over_bands() -> None:
    d = phase_diagram(
        base_config=santa_fe_config(seed=3),
        lam_values=(0.03, 0.12),
        theta_values=(0.02,),
        band_values=(3, 7),
        scan=SC,
    )
    assert d["n_bands"] == 2
    assert d["per_band"][0]["band"] == 3 and d["per_band"][1]["band"] == 7
    assert d["n_cells_total"] == 4


def test_diagram_rejects_unsorted_lams() -> None:
    with pytest.raises(ValueError):
        phase_diagram(
            base_config=santa_fe_config(),
            lam_values=(0.1, 0.01),
            theta_values=(0.02,),
            scan=SC,
        )


def test_diagram_rejects_bad_band() -> None:
    with pytest.raises(ValueError):
        phase_diagram(
            base_config=santa_fe_config(),
            lam_values=(0.05,),
            theta_values=(0.02,),
            band_values=(0,),
            scan=SC,
        )


# ---------------------------------------------------------------------------
# phase_feature_series + detect_phase_boundaries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_series() -> dict[str, Any]:
    flow = MarkovRegimeFlow(
        (RegimeState("calm", 1.0, 0.5), RegimeState("storm", 4.0, 0.5)),
        (0.98, 0.9),
        seed=12,
    )
    return phase_feature_series(santa_fe_config(seed=7), SC, bucket_seconds=10.0, flow=flow)


def test_feature_series_shapes(feature_series: dict[str, Any]) -> None:
    n = feature_series["n_buckets"]
    assert n == int(SC.horizon / 10.0)
    for ch in ("mid_sigma_ticks", "ofi", "mean_total_depth", "empty_frac", "n_mo_per_bucket"):
        assert len(feature_series[ch]) == n
    assert np.all(feature_series["empty_frac"] >= 0.0)
    assert np.all(feature_series["empty_frac"] <= 1.0)
    ofi = feature_series["ofi"][np.isfinite(feature_series["ofi"])]
    assert np.all(np.abs(ofi) <= 1.0)


def test_feature_series_deterministic() -> None:
    flow_a = MarkovRegimeFlow(
        (RegimeState("calm", 1.0, 0.5), RegimeState("storm", 4.0, 0.5)), (0.98, 0.9), seed=12
    )
    flow_b = MarkovRegimeFlow(
        (RegimeState("calm", 1.0, 0.5), RegimeState("storm", 4.0, 0.5)), (0.98, 0.9), seed=12
    )
    a = phase_feature_series(santa_fe_config(seed=7), SC, bucket_seconds=10.0, flow=flow_a)
    b = phase_feature_series(santa_fe_config(seed=7), SC, bucket_seconds=10.0, flow=flow_b)
    assert _blob_equal(a, b)


def test_feature_series_rejects_short_horizon() -> None:
    tiny = PhaseScanConfig(horizon=15.0, warmup=50.0, n_paths=2, seed=1)
    with pytest.raises(ValueError):
        phase_feature_series(santa_fe_config(seed=1), tiny, bucket_seconds=10.0)


def _planted_two_level() -> dict[str, Any]:
    rng = np.random.default_rng(99)
    return {
        "mid_sigma_ticks": np.concatenate(
            [0.02 + 0.005 * rng.standard_normal(30), 0.9 + 0.05 * rng.standard_normal(30)]
        ),
        "empty_frac": np.zeros(60),
    }


@pytest.mark.parametrize("method", ["optimal", "binseg", "bocpd"])
def test_boundaries_find_planted_jump(method: str) -> None:
    det = detect_phase_boundaries(_planted_two_level(), method=method, min_segment=3)
    assert det["method"] == method
    assert len(det["boundaries"]) >= 1
    assert all(0 < b < 60 for b in det["boundaries"])
    _assert_no_forbidden_keys(det)


def test_boundaries_coalesce_two_phase_labels() -> None:
    det = detect_phase_boundaries(_planted_two_level(), method="binseg", min_segment=3)
    phases = [s["phase"] for s in det["segments"]]
    assert det["n_segments"] == len(det["segments"])
    # Low-sigma block -> frozen, high-sigma block -> continuous.
    assert phases[0] == "frozen"
    assert phases[-1] == "continuous_liquidity"


def test_boundaries_constant_series_no_boundaries() -> None:
    feats = {"mid_sigma_ticks": np.full(30, 0.4), "empty_frac": np.zeros(30)}
    det = detect_phase_boundaries(feats, method="binseg")
    assert det["boundaries"] == []
    assert det["n_segments"] == 1


def test_boundaries_skip_metadata_and_arrays() -> None:
    """A bare ndarray input is treated as the mid_sigma channel itself."""
    arr = np.concatenate([np.full(20, 0.02), np.full(20, 0.9)])
    det = detect_phase_boundaries(arr, method="binseg", min_segment=3)
    assert len(det["boundaries"]) >= 1


def test_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError):
        detect_phase_boundaries({"mid_sigma_ticks": np.ones(3)}, method="binseg")
    with pytest.raises(ValueError):
        detect_phase_boundaries({"mid_sigma_ticks": np.ones(30)}, method="not_a_method")
    with pytest.raises(ValueError):
        detect_phase_boundaries(
            {"mid_sigma_ticks": np.ones(30), "ofi": np.ones(10)}, method="binseg"
        )
    with pytest.raises(ValueError):
        detect_phase_boundaries({}, method="binseg")


# ---------------------------------------------------------------------------
# phase_alarm — anytime-valid sequential detector on watch.py primitives
# ---------------------------------------------------------------------------


def test_alarm_fires_on_volatility_burst() -> None:
    rng = np.random.default_rng(21)
    series = np.concatenate([rng.normal(0.2, 0.05, 60), rng.normal(2.0, 0.3, 60)])
    al = phase_alarm(series, calibration_size=40, seed=5)
    assert al["alarm"] is True
    assert al["first_cross"] is not None
    assert al["wealth_final"] >= al["alarm_level"]
    assert al["shiryaev_roberts_final"] > 1.0
    _assert_no_forbidden_keys(al)


def test_alarm_quiet_on_exchangeable_series() -> None:
    rng = np.random.default_rng(22)
    al = phase_alarm(rng.normal(0.3, 0.06, 80), calibration_size=40, seed=6)
    assert al["alarm"] is False
    assert al["wealth_final"] < al["alarm_level"]


def test_alarm_deterministic() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(0.5, 0.4, 50)
    assert _blob_equal(phase_alarm(x, seed=9), phase_alarm(x, seed=9))


def test_alarm_fail_closed() -> None:
    with pytest.raises(ValueError):
        phase_alarm(np.array([0.1, 0.2, np.nan]), calibration_size=1)
    with pytest.raises(ValueError):
        phase_alarm(np.ones(20), calibration_size=20)
    with pytest.raises(ValueError):
        phase_alarm(np.ones(20), alpha=1.5)
    with pytest.raises(ValueError):
        phase_alarm(np.ones(20), calibration_size=1)


# ---------------------------------------------------------------------------
# impact_zscore / classify_impact_regime — closed-form Eq. (4)
# ---------------------------------------------------------------------------


def test_impact_zscore_closed_form() -> None:
    """Hand-computed Eq. 4: t=0 undefined (zero control dispersion),
    t=1 gives z = (3.5 - 1) / sqrt(0.7071 * 1.4142) = 2.5."""
    impacted = np.array([[1.0, 3.0], [2.0, 4.0]])
    control = np.array([[0.0, 0.0], [0.0, 2.0]])
    res = impact_zscore(impacted, control, min_defined=1)
    assert res["n_defined"] == 1
    assert math.isnan(res["z"][0])
    assert res["z"][1] == pytest.approx(2.5, abs=1e-12)
    assert res["mu_impacted"][1] == pytest.approx(3.5)


def test_impact_zscore_nan_paths_skipped() -> None:
    impacted = np.array([[1.0, 3.0], [np.nan, np.nan], [2.0, 4.0]])
    control = np.array([[0.0, 0.5], [0.0, 1.5]])
    res = impact_zscore(impacted, control, min_defined=1)
    assert math.isnan(res["z"][0])  # only one defined impacted path at t=0
    assert res["n_defined"] == 1


def test_impact_zscore_fail_closed() -> None:
    good = np.array([[1.0, 2.0], [1.5, 2.5]])
    with pytest.raises(ValueError):
        impact_zscore(np.array([1.0, 2.0]), good)  # 1-D
    with pytest.raises(ValueError):
        impact_zscore(good, np.array([[1.0]]))  # mismatched time axis
    with pytest.raises(ValueError):
        impact_zscore(good, np.array([[1.0, 2.0]]))  # single control path
    bad = good.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        impact_zscore(bad, good)
    with pytest.raises(ValueError):
        impact_zscore(np.full((3, 5), np.nan), good)  # no defined times


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (np.array([2.0, 1.0, 0.5, 0.2, 0.1]), "dissipative"),
        (np.array([0.5, 1.0, 2.0, 3.0, 4.0]), "non_dissipative"),
        (np.array([1.0, 1.2, 1.1, 1.05, 0.95]), "balanced"),
        (np.zeros(10), "dissipative"),
    ],
)
def test_classify_impact_regime(z: np.ndarray, expected: str) -> None:
    assert classify_impact_regime(z) == expected


def test_classify_impact_regime_fail_closed() -> None:
    with pytest.raises(ValueError):
        classify_impact_regime(np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        classify_impact_regime(np.array([1.0, 2.0, 3.0]), decay_frac=2.0)


def test_impact_regimes_cover_paper_labels() -> None:
    assert set(IMPACT_REGIMES) >= {"dissipative", "balanced", "non_dissipative"}


def test_impact_experiment_seeded() -> None:
    res = impact_zscore_experiment(
        config=santa_fe_config(seed=3),
        size=12,
        scan=PhaseScanConfig(horizon=160.0, warmup=100.0, n_paths=3, seed=3),
        n_samples=40,
    )
    assert res["impact_regime"] in IMPACT_REGIMES
    assert res["n_defined"] >= res["n_times"] // 2
    assert math.isfinite(res["z_peak"])
    assert res["claim"] == "simulator_internal_diagnostic_only"
    _assert_no_forbidden_keys(res)


def test_impact_experiment_deterministic() -> None:
    sc = PhaseScanConfig(horizon=160.0, warmup=100.0, n_paths=3, seed=3)
    a = impact_zscore_experiment(config=santa_fe_config(seed=3), size=12, scan=sc)
    b = impact_zscore_experiment(config=santa_fe_config(seed=3), size=12, scan=sc)
    assert _blob_equal(a, b)


def test_impact_experiment_fail_closed() -> None:
    with pytest.raises(ValueError):
        impact_zscore_experiment(config=santa_fe_config(), size=0, scan=SC)
    with pytest.raises(ValueError):
        impact_zscore_experiment(
            config=santa_fe_config(),
            size=5,
            scan=SC,
            side="up",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        impact_zscore_experiment(config={"lam": 0.06}, size=5, scan=SC)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# hysteresis_sweep — initial-condition dependence of the boundary
# ---------------------------------------------------------------------------


def test_hysteresis_sweep_structure() -> None:
    hs = hysteresis_sweep(
        base_config=santa_fe_config(seed=3),
        lam_values=(0.01, 0.03, 0.06, 0.12),
        scan=PhaseScanConfig(horizon=160.0, warmup=100.0, n_paths=3, seed=3),
    )
    assert len(hs["sigma_step_cold"]) == 4
    assert len(hs["collapse_hot"]) == 4
    assert hs["lam_c_cold"] >= 0.0 or math.isnan(hs["lam_c_cold"])
    assert hs["collapse_hot"][0] >= hs["collapse_hot"][-1]
    _assert_no_forbidden_keys(hs)


def test_hysteresis_deterministic() -> None:
    sc = PhaseScanConfig(horizon=120.0, warmup=80.0, n_paths=2, seed=5)
    a = hysteresis_sweep(base_config=santa_fe_config(seed=5), lam_values=(0.03, 0.12), scan=sc)
    b = hysteresis_sweep(base_config=santa_fe_config(seed=5), lam_values=(0.03, 0.12), scan=sc)
    assert _blob_equal(a, b)


def test_hysteresis_rejects_bad_lams() -> None:
    with pytest.raises(ValueError):
        hysteresis_sweep(base_config=santa_fe_config(), lam_values=(), scan=SC)


# ---------------------------------------------------------------------------
# cluster_injection_phases — deterministic 2-means on phase observables
# ---------------------------------------------------------------------------


def _ob(sigma: float, empty: float) -> dict[str, float]:
    return {"sigma_step_ticks": sigma, "empty_time_frac": empty}


def test_cluster_separates_three_phases() -> None:
    obs = (
        [_ob(float("nan"), 0.9), _ob(0.4, 0.8), _ob(0.35, 0.7)]
        + [_ob(0.01, 0.0), _ob(0.02, 0.0), _ob(0.015, 0.0)]
        + [_ob(0.5, 0.0), _ob(0.6, 0.05), _ob(0.55, 0.0)]
    )
    agg = cluster_injection_phases(obs)
    labs = agg["phase_labels"]
    assert labs[:3] == ["collapsed"] * 3
    assert set(labs[3:6]) == {"frozen"}
    assert set(labs[6:]) == {"continuous_liquidity"}
    assert agg["counts"]["collapsed"] == 3
    assert agg["n_collapsed_direct"] == 3


def test_cluster_deterministic() -> None:
    obs = [_ob(0.01, 0.0), _ob(0.02, 0.0), _ob(0.5, 0.0), _ob(0.6, 0.0), _ob(0.4, 0.0)]
    a = cluster_injection_phases(obs)
    b = cluster_injection_phases(obs)
    assert a == b


def test_cluster_fail_closed() -> None:
    with pytest.raises(ValueError):
        cluster_injection_phases([_ob(0.1, 0.0), _ob(0.2, 0.0)])
    with pytest.raises(ValueError):
        cluster_injection_phases([{"empty_time_frac": 0.0}] * 3)
    with pytest.raises(ValueError):
        cluster_injection_phases([_ob(0.1, float("nan"))] * 3)
    with pytest.raises(ValueError):
        cluster_injection_phases([_ob(0.5, 0.0)] * 4)  # constant sigma


# ---------------------------------------------------------------------------
# bench — flat dict, deterministic, seconds-scale
# ---------------------------------------------------------------------------


def test_bench_flat_float_dict() -> None:
    out = bench_agentic_lob()
    assert len(out) >= 10
    assert all(isinstance(k, str) for k in out)
    assert all(isinstance(v, float) for v in out.values())
    assert out["diagram_n_cells"] == 9.0
    assert out["boundaries_n"] >= 1.0
    assert out["alarm_fired"] == 1.0
    _assert_no_forbidden_keys(out)


def test_bench_deterministic() -> None:
    assert bench_agentic_lob(seed=13) == bench_agentic_lob(seed=13)


def test_bench_rejects_bool_seed() -> None:
    with pytest.raises(ValueError):
        bench_agentic_lob(seed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Revision + blob honesty sweep
# ---------------------------------------------------------------------------


def test_revision_and_labels_everywhere(
    starved_cell: dict[str, Any],
    santafe_cell: dict[str, Any],
    small_diagram: dict[str, Any],
) -> None:
    assert starved_cell["data_source"] == AGENTIC_LOB_REVISION
    assert starved_cell["label"] == "SYNTHETIC"
    assert santafe_cell["label"] == "SYNTHETIC"
    assert small_diagram["label"] == "SYNTHETIC"
    _assert_no_forbidden_keys(starved_cell)
    _assert_no_forbidden_keys(santafe_cell)
