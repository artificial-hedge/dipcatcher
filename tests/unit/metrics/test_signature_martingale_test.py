"""Signature-martingale test — SYNTHETIC correctness tests.

Validates the wave-22 expected-signature martingale layer: windowed
signature-moment features (Chevyrev & Oberhauser 2018/2022,
arXiv:1810.10971) composed through ``models.path_signatures.signature``,
the Hotelling + Rademacher wild-bootstrap calibration, the within-window
increment-permutation ordering test, the Vovk & Wang
(arXiv:1912.06116) e-process over window boundaries, the orchestrator
blob, the SYNTHETIC planted worlds and the bench. All ensembles are
seeded SYNTHETIC paths — correctness evidence only, never market
evidence (AGENTS.md honesty contract #2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.signature_martingale_test import (
    DEFAULT_LAM,
    DEFAULT_N_RESAMPLES,
    DEFAULT_WINDOW,
    E_MODES,
    FEATURE_NAMES,
    KIND,
    MIN_WINDOW,
    MIN_WINDOWS,
    SCHEMA,
    bench_signature_martingale_test,
    expected_signature_test,
    martingale_e_process,
    ordering_permutation_test,
    signature_martingale_test,
    simulate_martingale_paths,
    time_augmented_path,
    window_signature_features,
)
from quant_fund.models.path_signatures import signature
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent

Array = np.ndarray

N_MC_SIZE = 24
N_MC_POWER = 16


def _bm_path(seed: int, n: int = 336, sigma: float = 1.0) -> Array:
    return np.asarray(simulate_martingale_paths("brownian", n - 1, seed, sigma=sigma)["path"])


def _world(kind: str, seed: int, n: int = 336, **kwargs: float) -> Array:
    return np.asarray(simulate_martingale_paths(kind, n - 1, seed, **kwargs)["path"])


def _rejection_rate(
    kind: str,
    arm: str,
    alpha: float = 0.05,
    n_mc: int = N_MC_SIZE,
    seed: int = 11,
    n: int = 336,
    window: int = DEFAULT_WINDOW,
    **kwargs: float,
) -> float:
    hits = []
    for rep in range(n_mc):
        path = _world(kind, seed + 7919 * rep, n, **kwargs)
        res = signature_martingale_test(
            path, window, n_boot=99, n_perm=99, seed=seed + 104729 * rep
        )
        hits.append(float(res[arm]) <= alpha)
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# time_augmented_path
# ---------------------------------------------------------------------------


def test_time_augmented_path_shape_and_clock() -> None:
    x = np.array([1.0, 2.0, 0.5, 3.0])
    p = time_augmented_path(x)
    assert p.shape == (4, 2)
    np.testing.assert_allclose(p[:, 0], [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(p[:, 1], x)


def test_time_augmented_path_accepts_iterable() -> None:
    p = time_augmented_path([0.0, 1.0])
    np.testing.assert_allclose(p, [[0.0, 0.0], [1.0, 1.0]])


def test_time_augmented_path_fail_closed() -> None:
    with pytest.raises(ValueError):
        time_augmented_path(np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        time_augmented_path(np.ones((3, 3)))


# ---------------------------------------------------------------------------
# window_signature_features
# ---------------------------------------------------------------------------


def test_window_features_shapes_and_names() -> None:
    out = window_signature_features(_bm_path(0, n=97), window=8)
    assert out["z"].shape == (12, 3)
    assert out["sigma_hat"].shape == (12,)
    assert out["increments"].shape == (96,)
    assert out["n_windows"] == 12
    assert out["window"] == 8
    assert out["feature_names"] == FEATURE_NAMES


def test_window_features_drops_tail() -> None:
    out = window_signature_features(_bm_path(0, n=100), window=8)
    # 99 increments -> 12 complete windows of 8, 3 dropped
    assert out["n_windows"] == 12
    assert out["dropped_tail_increments"] == 3


def test_window_features_z1_identity() -> None:
    path = _bm_path(1, n=97)
    out = window_signature_features(path, window=8)
    dx = np.diff(path)[:96].reshape(12, 8)
    z1_expected = dx.sum(axis=1) / (out["sigma_hat"] * math.sqrt(8))
    np.testing.assert_allclose(out["z"][:, 0], z1_expected, rtol=1e-12)


def test_window_features_z2_matches_signature_solver() -> None:
    """Pin the composed-solver identity: z2 comes from Sig^2_{u,X}."""
    path = _bm_path(2, n=65)
    w = 8
    out = window_signature_features(path, window=w)
    mids = (np.arange(w) + 0.5) / w
    for i in range(out["n_windows"]):
        seg = path[i * w : (i + 1) * w + 1]
        sig = signature(time_augmented_path(seg), order=2)
        # flat word order d=2: [Sig1_u, Sig1_X, Sig2_uu, Sig2_uX, Sig2_Xu, Sig2_XX]
        sig2_ux = float(sig[3])
        dx_w = np.diff(seg)
        np.testing.assert_allclose(sig2_ux, float(mids @ dx_w), atol=1e-12)
        z2_expected = sig2_ux / (out["sigma_hat"][i] * math.sqrt((4 * w * w - 1) / (12 * w)))
        np.testing.assert_allclose(out["z"][i, 1], z2_expected, rtol=1e-12)


def test_window_features_z3_identity() -> None:
    path = _bm_path(3, n=97)
    out = window_signature_features(path, window=8)
    dx = np.diff(path)[:96].reshape(12, 8)
    z3_expected = (dx[:, :-1] * dx[:, 1:]).sum(axis=1) / (out["sigma_hat"] ** 2 * math.sqrt(7))
    np.testing.assert_allclose(out["z"][:, 2], z3_expected, rtol=1e-12)


def test_shuffle_product_identity() -> None:
    """Shuffle product: Sig^2_{u,X} + Sig^2_{X,u} == Sig^1_u * Sig^1_X == ΔX."""
    path = _bm_path(4, n=33)
    w = 8
    seg = path[: w + 1]
    sig = signature(time_augmented_path(seg), order=2)
    np.testing.assert_allclose(sig[3] + sig[4], seg[-1] - seg[0], atol=1e-12)


def test_window_features_null_moments_vanish() -> None:
    """Expected-signature claim: under the martingale null E[z_w] ~ 0."""
    rng = np.random.default_rng(5)
    z_rows = []
    for _rep in range(40):
        path = np.concatenate([[0.0], np.cumsum(rng.standard_normal(335))])
        z_rows.append(window_signature_features(path, 16)["z"])
    z = np.concatenate(z_rows)
    means = z.mean(axis=0)
    ses = z.std(axis=0, ddof=1) / math.sqrt(z.shape[0])
    np.testing.assert_array_less(np.abs(means), 4.0 * ses)
    # unit marginal variance claim (moderately loose for the plug-in scale)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), np.ones(3), rtol=0.35)


def test_window_features_fail_closed() -> None:
    with pytest.raises(ValueError):
        window_signature_features(np.arange(3.0), window=8)
    with pytest.raises(ValueError):
        window_signature_features(_bm_path(0, n=50), window=3)  # < MIN_WINDOW
    with pytest.raises(ValueError):
        window_signature_features(_bm_path(0, n=50), window=True)
    with pytest.raises(ValueError):
        window_signature_features(np.ones((40, 2)), window=8)


def test_window_features_degenerate_window_raises() -> None:
    path = _bm_path(0, n=97)
    path[8:17] = path[8]  # flat second window -> σ̂_w = 0
    with pytest.raises(ValueError, match="degenerate window"):
        window_signature_features(path, window=8)


# ---------------------------------------------------------------------------
# expected_signature_test (Hotelling + wild bootstrap)
# ---------------------------------------------------------------------------


def test_expected_signature_output_contract() -> None:
    out = expected_signature_test(_bm_path(6), window=16, n_boot=99, seed=0)
    assert out["t_stat"] >= 0.0
    assert 0.0 < out["p_chi2"] <= 1.0
    assert 0.0 < out["p_boot"] <= 1.0
    assert out["df"] == 3.0
    assert out["z_bar"].shape == (3,)
    assert out["feature_cov"].shape == (3, 3)
    assert out["calibration"] == "rademacher_wild_bootstrap"


def test_expected_signature_deterministic() -> None:
    path = _bm_path(7)
    a = expected_signature_test(path, 16, n_boot=149, seed=17)
    b = expected_signature_test(path, 16, n_boot=149, seed=17)
    assert a["p_boot"] == b["p_boot"]
    assert a["t_stat"] == b["t_stat"]


def test_expected_signature_p_grid() -> None:
    """Bootstrap p-values live on the (k+1)/(B+1) grid; chi2 reference is
    seed-independent (deterministic function of the path)."""
    path = _bm_path(8)
    a = expected_signature_test(path, 16, n_boot=149, seed=17)
    b = expected_signature_test(path, 16, n_boot=149, seed=18)
    assert a["p_chi2"] == b["p_chi2"]
    assert a["p_boot"] * 150 == pytest.approx(round(a["p_boot"] * 150))


def test_expected_signature_detects_planted_drift() -> None:
    path = _world("drift", 9, drift=0.30)
    out = expected_signature_test(path, 16, n_boot=199, seed=0)
    assert out["p_boot"] <= 0.05
    assert out["p_chi2"] <= 0.05


def test_expected_signature_size_on_brownian_null() -> None:
    rate = _rejection_rate("brownian", "p_boot", alpha=0.05)
    assert rate <= 0.25  # nominal 0.05; bound guards gross miscalibration only


def test_expected_signature_fail_closed() -> None:
    path = _bm_path(10)
    with pytest.raises(ValueError):
        expected_signature_test(path, 16, n_boot=10)
    with pytest.raises(ValueError):
        expected_signature_test(path, 16, n_boot=True)
    with pytest.raises(ValueError):
        expected_signature_test(np.array([1.0, np.inf]), 16)


# ---------------------------------------------------------------------------
# ordering_permutation_test
# ---------------------------------------------------------------------------


def test_ordering_output_contract() -> None:
    out = ordering_permutation_test(_bm_path(11), window=16, n_perm=99, seed=0)
    assert 0.0 < out["p_perm"] <= 1.0
    assert math.isfinite(out["s_stat"])
    assert math.isfinite(out["s_perm_center"])
    assert out["calibration"] == "within_window_increment_permutation"


def test_ordering_deterministic() -> None:
    path = _bm_path(12)
    a = ordering_permutation_test(path, 16, n_perm=149, seed=3)
    b = ordering_permutation_test(path, 16, n_perm=149, seed=3)
    assert a["p_perm"] == b["p_perm"] and a["s_stat"] == b["s_stat"]


def test_ordering_blind_to_pure_drift() -> None:
    """Theory-pinned property: permutation preserves each window's increment
    multiset, so a uniform drift is (near-)invariant — the ordering arm must
    stay near its null even under a strong planted trend."""
    rate = _rejection_rate("drift", "p_perm", alpha=0.05, n_mc=N_MC_POWER, drift=0.30)
    assert rate <= 0.35


def test_ordering_detects_ou_mean_reversion() -> None:
    rate = _rejection_rate("ou", "p_perm", alpha=0.10, n_mc=N_MC_POWER, theta=0.30)
    assert rate >= 0.25


def test_ordering_detects_ar1_momentum() -> None:
    rate = _rejection_rate("ar1", "p_perm", alpha=0.05, n_mc=N_MC_POWER, phi=0.30)
    assert rate >= 0.5


def test_ordering_size_on_brownian_null() -> None:
    rate = _rejection_rate("brownian", "p_perm", alpha=0.05)
    assert rate <= 0.25


def test_ordering_fail_closed() -> None:
    path = _bm_path(13)
    with pytest.raises(ValueError):
        ordering_permutation_test(path, 16, n_perm=5)
    flat = _bm_path(13, n=97)
    flat[8:17] = flat[8]
    with pytest.raises(ValueError, match="degenerate window"):
        ordering_permutation_test(flat, 8, n_perm=99)


# ---------------------------------------------------------------------------
# martingale_e_process
# ---------------------------------------------------------------------------


def test_e_process_trajectory_contract() -> None:
    out = martingale_e_process(_bm_path(14), 16)
    n_w = out["n_windows"]
    assert out["e"].shape == (n_w,)
    assert out["e"][0] > 0.0
    np.testing.assert_allclose(out["e_final"], float(out["e"][-1]))
    assert out["threshold"] == pytest.approx(1.0 / 0.05)
    assert out["s_margin"].shape == (n_w,)
    assert out["e_mode"] == "gaussian_lr"
    assert out["lam"] == DEFAULT_LAM


def test_e_process_log_cumsum_consistency() -> None:
    path = _bm_path(15)
    out = martingale_e_process(path, 16, lam=0.4)
    s = out["s_margin"]
    factors = np.cosh(0.4 * s) * math.exp(-0.5 * 0.4 * 0.4)
    np.testing.assert_allclose(out["e"], np.cumprod(factors), rtol=1e-10)


def test_e_process_bounded_mode_unit_mean_shape() -> None:
    out = martingale_e_process(_bm_path(16), 16, e_mode="bounded", lam=0.5)
    assert out["e_mode"] == "bounded"
    assert np.all(out["e"] > 0.0)
    # bounded factor e_w = 1 + lam tanh(s) lies in (1-lam, 1+lam)
    assert np.all(out["e"] <= (1.0 + 0.5) ** out["n_windows"])


def test_e_process_grows_under_drift() -> None:
    null_med = np.median(
        [martingale_e_process(_world("brownian", 20 + r), 16)["e_final"] for r in range(12)]
    )
    drift_med = np.median(
        [
            martingale_e_process(_world("drift", 120 + r, drift=0.20), 16)["e_final"]
            for r in range(12)
        ]
    )
    assert drift_med > null_med


def test_e_process_ville_conservative_on_null() -> None:
    hits = [martingale_e_process(_world("brownian", 300 + r), 16)["reject"] for r in range(20)]
    assert np.mean(hits) <= 0.1


def test_e_process_fail_closed() -> None:
    path = _bm_path(17)
    with pytest.raises(ValueError):
        martingale_e_process(path, 16, e_mode="bogus")
    with pytest.raises(ValueError):
        martingale_e_process(path, 16, lam=0.0)
    with pytest.raises(ValueError):
        martingale_e_process(path, 16, lam=1.0)
    with pytest.raises(ValueError):
        martingale_e_process(path, 16, alpha=0.0)
    with pytest.raises(ValueError):
        martingale_e_process(path, 16, alpha=1.0)
    assert all(m in E_MODES for m in ("gaussian_lr", "bounded"))


# ---------------------------------------------------------------------------
# signature_martingale_test orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_blob_contract() -> None:
    out = signature_martingale_test(_bm_path(18), 16, n_boot=99, n_perm=99, seed=0)
    assert out["schema"] == SCHEMA
    assert out["kind"] == KIND
    assert out["claim"] == "research_only"
    assert out["live_pnl_claim"] is False
    assert out["citation"].startswith("Chevyrev")
    assert out["feature_names"] == FEATURE_NAMES
    for key in (
        "t_stat",
        "p_chi2",
        "p_boot",
        "s_stat",
        "p_perm",
        "p_omnibus",
        "e_final",
        "e_first_cross",
        "e_reject",
        "e_threshold",
        "reject_omnibus",
        "reject_boot",
        "reject_perm",
        "reject_chi2",
    ):
        assert key in out


def test_orchestrator_omnibus_and_flags() -> None:
    out = signature_martingale_test(_bm_path(19), 16, n_boot=99, n_perm=99, seed=1)
    expected = min(1.0, 2.0 * min(out["p_boot"], out["p_perm"]))
    assert out["p_omnibus"] == pytest.approx(expected)
    assert out["reject_omnibus"] == (out["p_omnibus"] <= out["alpha"])
    assert out["reject_boot"] == (out["p_boot"] <= out["alpha"])
    assert out["reject_perm"] == (out["p_perm"] <= out["alpha"])
    assert out["reject_chi2"] == (out["p_chi2"] <= out["alpha"])


def test_orchestrator_deterministic_end_to_end() -> None:
    path = _bm_path(20)
    a = signature_martingale_test(path, 16, n_boot=149, n_perm=149, seed=23)
    b = signature_martingale_test(path, 16, n_boot=149, n_perm=149, seed=23)
    for key in ("p_boot", "p_perm", "p_omnibus", "e_final", "t_stat", "s_stat"):
        assert a[key] == b[key]


def test_orchestrator_rejects_planted_drift() -> None:
    out = signature_martingale_test(
        _world("drift", 21, drift=0.30), 16, n_boot=199, n_perm=199, seed=0
    )
    assert out["reject_omnibus"] is True
    assert out["reject_boot"] is True


def test_orchestrator_size_on_brownian_null() -> None:
    rate = _rejection_rate("brownian", "p_omnibus", alpha=0.05, n_mc=N_MC_SIZE)
    assert rate <= 0.25


def test_orchestrator_size_at_010_and_garch_null() -> None:
    rate10 = _rejection_rate("brownian", "p_omnibus", alpha=0.10, n_mc=N_MC_SIZE)
    assert rate10 <= 0.35
    rate_g = _rejection_rate(
        "garch", "p_omnibus", alpha=0.05, n_mc=N_MC_SIZE, alpha_garch=0.12, beta_garch=0.82
    )
    assert rate_g <= 0.30


def test_orchestrator_no_forbidden_metric_keys() -> None:
    out = signature_martingale_test(_bm_path(22), 16, n_boot=49, n_perm=49, seed=0)
    assert family_blob_forbidden_metrics_absent(out)


# ---------------------------------------------------------------------------
# simulate_martingale_paths (SYNTHETIC worlds)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["brownian", "garch", "drift", "ou", "ar1"])
def test_simulate_worlds_shapes(kind: str) -> None:
    out = simulate_martingale_paths(kind, 100, 5)
    assert out["increments"].shape == (100,)
    assert out["path"].shape == (101,)
    assert out["path"][0] == 0.0
    np.testing.assert_allclose(out["path"], np.concatenate([[0.0], np.cumsum(out["increments"])]))
    assert out["kind"] == kind
    assert out["config"]["data_label"] == "SYNTHETIC"


def test_simulate_deterministic_bit_identical() -> None:
    a = simulate_martingale_paths("garch", 80, 9)
    b = simulate_martingale_paths("garch", 80, 9)
    np.testing.assert_array_equal(a["increments"], b["increments"])
    c = simulate_martingale_paths("garch", 80, 10)
    assert not np.array_equal(a["increments"], c["increments"])


def test_simulate_garch_vol_clustering() -> None:
    dx = simulate_martingale_paths("garch", 4000, 11, alpha_garch=0.12, beta_garch=0.82)[
        "increments"
    ]
    sq = dx * dx
    lag1 = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    assert lag1 > 0.05  # volatility clusters: squared increments autocorrelate
    assert np.corrcoef(dx[:-1], dx[1:])[0, 1] < 0.05  # increments stay ~unpredictable


def test_simulate_ar1_momentum_planted() -> None:
    dx = simulate_martingale_paths("ar1", 4000, 12, phi=0.40)["increments"]
    assert np.corrcoef(dx[:-1], dx[1:])[0, 1] == pytest.approx(0.40, abs=0.05)


def test_simulate_ou_mean_reversion_planted() -> None:
    out = simulate_martingale_paths("ou", 4000, 13, theta=0.20)
    dx, path = out["increments"], out["path"]
    # dx_t = -θ x_{t-1} + ε: correlation of increment vs lagged level ~ -θ·sd(x)
    corr = np.corrcoef(path[1:-1], dx[1:])[0, 1]
    assert corr < -0.1


def test_simulate_fail_closed() -> None:
    with pytest.raises(ValueError):
        simulate_martingale_paths("bogus", 100, 0)
    with pytest.raises(ValueError):
        simulate_martingale_paths("brownian", 7, 0)
    with pytest.raises(ValueError):
        simulate_martingale_paths("brownian", 100, 0, sigma=0.0)
    with pytest.raises(ValueError):
        simulate_martingale_paths("ar1", 100, 0, phi=1.0)
    with pytest.raises(ValueError):
        simulate_martingale_paths("ou", 100, 0, theta=-0.1)
    with pytest.raises(ValueError):
        simulate_martingale_paths("garch", 100, 0, alpha_garch=0.6, beta_garch=0.6)


# ---------------------------------------------------------------------------
# bench_signature_martingale_test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_blob() -> dict[str, float]:
    return bench_signature_martingale_test(7)


def test_bench_contract_all_synthetic_floats(bench_blob: dict[str, float]) -> None:
    assert bench_blob, "bench returned empty blob"
    for key, value in bench_blob.items():
        assert key.startswith("synthetic_"), key
        assert isinstance(value, float), key
        assert math.isfinite(value), key


def test_bench_size_holds(bench_blob: dict[str, float]) -> None:
    assert bench_blob["synthetic_size_bm_005"] <= 0.30
    assert bench_blob["synthetic_size_bm_010"] <= 0.40
    assert bench_blob["synthetic_size_garch_005"] <= 0.35
    assert bench_blob["synthetic_size_bm_perm_005"] <= 0.30
    assert bench_blob["synthetic_ville_rate_null_005"] <= 0.15


def test_bench_power_monotone_in_drift(bench_blob: dict[str, float]) -> None:
    assert (
        bench_blob["synthetic_power_drift_0p05_005"]
        < bench_blob["synthetic_power_drift_0p10_005"]
        < bench_blob["synthetic_power_drift_0p20_005"]
    )
    assert bench_blob["synthetic_power_drift_0p20_005"] >= 0.4


def test_bench_power_ordering_alternatives(bench_blob: dict[str, float]) -> None:
    assert bench_blob["synthetic_power_ou_t0p10_005"] < bench_blob["synthetic_power_ou_t0p30_005"]
    assert bench_blob["synthetic_power_ar1_phi0p30_005"] >= 0.6
    assert bench_blob["synthetic_power_ar1_perm_phi0p30_005"] >= 0.4


def test_bench_evalue_growth_under_drift(bench_blob: dict[str, float]) -> None:
    assert bench_blob["synthetic_eval_drift_hi_median"] > bench_blob["synthetic_eval_null_median"]


def test_bench_honesty_flags(bench_blob: dict[str, float]) -> None:
    assert bench_blob["synthetic_live_pnl_claim"] == 0.0
    assert bench_blob["synthetic_n_mc"] > 0
    assert bench_blob["synthetic_n_boot"] <= DEFAULT_N_RESAMPLES
    assert bench_blob["synthetic_window"] == float(DEFAULT_WINDOW)
    assert int(bench_blob["synthetic_window"]) >= MIN_WINDOW * MIN_WINDOWS


def test_bench_deterministic() -> None:
    a = bench_signature_martingale_test(13)
    b = bench_signature_martingale_test(13)
    assert a == b
