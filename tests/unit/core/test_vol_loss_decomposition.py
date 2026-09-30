"""Loss-choice vs model-choice decomposition — SYNTHETIC correctness tests.

Validates the mechanism of Tokajuk & Chudziak (2026), "Loss Choice or Model
Choice? The Role of Forecast Level in Cryptocurrency Volatility Forecasting",
arXiv:2609.27024 (ADMA 2026), on a seeded planted GARCH(1,1) world
(AGENTS.md honesty contract #2): raw score variation is loss-dominated
(forecast LEVEL), aligned variation is model-dominated (day-to-day MOVEMENTS),
one-day VaR breach-rate spread narrows after validation-based alignment, and
the alignment constant is fitted on validation only (fail-closed overlap
guard; perturbing test data cannot move it). Research-diagnostic only —
live_pnl_claim=False; never market evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_fund.metrics.vol_loss_decomposition import (
    DEFAULT_LEVEL_FACTORS,
    LossModelWorld,
    gaussian_var_threshold,
    level_share,
    loss_model_decomposition,
    pairwise_marginal_gaps,
    run_loss_model_decomposition_study,
    simulate_loss_model_world,
    two_way_score_shares,
    validation_level_alignment,
    var_breach_rates,
)
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

# Study seeds deliberately outside the calibration set (0-7) used while tuning
# the planted world; margins below are wide against every observed seed.
STUDY_SEEDS = (11, 12, 13)


def _windows(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint validation (first half) / test (second half) boolean masks."""
    val = np.zeros(n, dtype=bool)
    val[: n // 2] = True
    test = np.zeros(n, dtype=bool)
    test[n // 2 :] = True
    return val, test


def _toy_series(n: int = 40, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Positive heteroskedastic proxy and a mis-leveled positive forecast."""
    rng = np.random.default_rng(seed)
    proxy = np.exp(rng.normal(0.0, 0.5, n)) + 0.1
    forecast = 0.4 * proxy * np.exp(rng.normal(0.0, 0.2, n))
    return proxy, forecast


# ---------------------------------------------------------------------------
# validation_level_alignment — paper Eq. (1)
# ---------------------------------------------------------------------------


def test_alignment_constant_matches_closed_form() -> None:
    proxy, forecast = _toy_series()
    val, test = _windows()
    fit = validation_level_alignment(
        forecast, proxy, validation_mask=val, test_mask=test, mode="multiplicative"
    )
    expected_c = float(np.mean(proxy[val] / forecast[val]))
    assert fit["constant"] == pytest.approx(expected_c, rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(fit["aligned_test"]), expected_c * forecast[test], rtol=1e-12
    )
    # Validation rows are left raw — the paper applies c to TEST forecasts.
    np.testing.assert_allclose(np.asarray(fit["aligned"])[val], forecast[val], rtol=0, atol=0)


def test_alignment_constant_minimizes_validation_qlike() -> None:
    # Eq. (1) claim: c = mean(h/f) minimizes validation QLIKE over constant
    # rescalings — verified on a grid around the fitted constant.
    proxy, forecast = _toy_series()
    val, test = _windows()
    fit = validation_level_alignment(
        forecast, proxy, validation_mask=val, test_mask=test, mode="multiplicative"
    )
    c = float(fit["constant"])
    assert float(fit["validation_qlike_aligned"]) <= float(fit["validation_qlike_raw"]) + 1e-12
    for mult in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
        alt = float(
            np.mean(
                proxy[val] / (mult * c * forecast[val])
                - np.log(proxy[val] / (mult * c * forecast[val]))
                - 1.0
            )
        )
        assert float(fit["validation_qlike_aligned"]) < alt


def test_additive_variant_minimizes_validation_mse() -> None:
    proxy, forecast = _toy_series()
    val, test = _windows()
    fit = validation_level_alignment(
        forecast, proxy, validation_mask=val, test_mask=test, mode="additive"
    )
    b = float(np.mean(proxy[val] - forecast[val]))
    assert fit["constant"] == pytest.approx(b, rel=1e-12)
    np.testing.assert_allclose(np.asarray(fit["aligned_test"]), forecast[test] + b, rtol=1e-12)
    assert float(fit["validation_mse_aligned"]) <= float(fit["validation_mse_raw"]) + 1e-12


def test_overlap_guard_fails_closed() -> None:
    # The lookahead guard: shuffling test dates into the validation window
    # must raise, never silently fit on test data.
    proxy, forecast = _toy_series()
    val, test = _windows()
    overlapped_val = val.copy()
    overlapped_val[np.flatnonzero(test)[:5]] = True
    with pytest.raises(ValueError, match="overlap"):
        validation_level_alignment(forecast, proxy, validation_mask=overlapped_val, test_mask=test)
    fully_shared = test.copy()
    with pytest.raises(ValueError, match="overlap"):
        validation_level_alignment(forecast, proxy, validation_mask=fully_shared, test_mask=test)


def test_alignment_constant_never_sees_test_window() -> None:
    # No-lookahead evidence: rewriting the TEST-window proxy (and returns)
    # leaves the constant and aligned forecasts untouched; rewriting the
    # VALIDATION proxy changes them.
    proxy, forecast = _toy_series()
    val, test = _windows()
    base = validation_level_alignment(forecast, proxy, validation_mask=val, test_mask=test)
    mutated = proxy.copy()
    mutated[test] *= 7.5
    same = validation_level_alignment(forecast, mutated, validation_mask=val, test_mask=test)
    assert same["constant"] == pytest.approx(float(base["constant"]), rel=0, abs=0)
    np.testing.assert_array_equal(
        np.asarray(same["aligned_test"]), np.asarray(base["aligned_test"])
    )
    mutated_val = proxy.copy()
    mutated_val[val] *= 2.0
    changed = validation_level_alignment(forecast, mutated_val, validation_mask=val, test_mask=test)
    assert changed["constant"] != pytest.approx(float(base["constant"]), rel=1e-6)
    # The validation constant honestly differs from the test-oracle constant
    # (sampling noise) — the guard has teeth.
    oracle_c = float(np.mean(proxy[test] / forecast[test]))
    assert float(base["constant"]) != pytest.approx(oracle_c, rel=1e-6)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f, h, val, test: (f, h, val.astype(int), test),
        lambda f, h, val, test: (f, h, val[:-1], test[:-1]),
        lambda f, h, val, test: (f, h, np.zeros_like(val), test),
        lambda f, h, val, test: (f, h, val, np.zeros_like(test)),
        lambda f, h, val, test: (-f, h, val, test),
        lambda f, h, val, test: (f, -h, val, test),
        lambda f, h, val, test: (np.full_like(f, np.nan), h, val, test),
    ],
)
def test_alignment_fail_closed_edges(mutate: Any) -> None:
    proxy, forecast = _toy_series()
    val, test = _windows()
    f, h, v, t = mutate(forecast, proxy, val, test)
    with pytest.raises(ValueError):
        validation_level_alignment(f, h, validation_mask=v, test_mask=t)
    with pytest.raises(ValueError):
        validation_level_alignment(
            forecast, proxy, validation_mask=val, test_mask=test, mode="bogus"
        )


def test_additive_shift_breaking_positivity_fails_closed() -> None:
    n = 40
    val, test = _windows(n)
    forecast = np.full(n, 2.0)
    proxy = np.zeros(n)  # b = mean(0 - 2) = -2 → aligned test forecast = 0
    with pytest.raises(ValueError, match="non-positive"):
        validation_level_alignment(
            forecast, proxy, validation_mask=val, test_mask=test, mode="additive"
        )


# ---------------------------------------------------------------------------
# level_share — paper §3.2 RQ2
# ---------------------------------------------------------------------------


def test_level_share_pure_offset_is_one() -> None:
    rng = np.random.default_rng(5)
    f = np.exp(rng.normal(0.0, 0.4, 200))
    assert level_share(2.5 * f, f) == pytest.approx(1.0, abs=1e-12)


def test_level_share_mean_zero_movements_is_zero() -> None:
    rng = np.random.default_rng(6)
    f = np.exp(rng.normal(0.0, 0.3, 400))
    d = np.array([0.1, -0.1] * 200)  # exactly zero-mean log movement
    g = f * np.exp(d)
    assert level_share(g, f) == pytest.approx(0.0, abs=1e-12)


def test_level_share_fail_closed_edges() -> None:
    f = np.ones(50)
    with pytest.raises(ValueError, match="undefined"):
        level_share(f, f.copy())
    with pytest.raises(ValueError):
        level_share(-f, f)
    with pytest.raises(ValueError):
        level_share(f[:-1], f)


# ---------------------------------------------------------------------------
# pairwise_marginal_gaps / two_way_score_shares / loss_model_decomposition
# ---------------------------------------------------------------------------


def test_pairwise_gaps_hand_computable_2x2() -> None:
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])  # loss marginals 1.5, 3.5; model 2.0, 3.0
    gaps = pairwise_marginal_gaps(scores)
    assert gaps["delta_loss"] == pytest.approx(2.0)
    assert gaps["delta_model"] == pytest.approx(1.0)
    assert gaps["loss_to_model_ratio"] == pytest.approx(2.0)
    assert gaps["n_loss_pairs"] == 1 and gaps["n_model_pairs"] == 1


def test_pairwise_gaps_symmetric_matrix_ratio_one() -> None:
    scores = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    gaps = pairwise_marginal_gaps(scores)
    assert gaps["delta_loss"] == pytest.approx(gaps["delta_model"])
    assert gaps["loss_to_model_ratio"] == pytest.approx(1.0)
    assert gaps["n_loss_pairs"] == 3 and gaps["n_model_pairs"] == 3


def test_pairwise_gaps_degenerate_and_malformed() -> None:
    with pytest.raises(ValueError, match="no marginal variation"):
        pairwise_marginal_gaps(np.ones((3, 2)))
    # Δ_M = 0 with Δ_L > 0 is honestly infinite (loss-dominated).
    gaps = pairwise_marginal_gaps(np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]]))
    assert gaps["delta_model"] == 0.0
    assert gaps["loss_to_model_ratio"] == float("inf")
    with pytest.raises(ValueError):
        pairwise_marginal_gaps(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        pairwise_marginal_gaps(np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        pairwise_marginal_gaps(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_two_way_shares_hand_computable() -> None:
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    shares = two_way_score_shares(scores)
    # grand 2.5; ss_tot 5; ss_loss 4; ss_model 1; residual 0.
    assert shares["loss_share"] == pytest.approx(0.8)
    assert shares["model_share"] == pytest.approx(0.2)
    assert shares["residual_share"] == pytest.approx(0.0, abs=1e-12)
    assert shares["loss_share"] + shares["model_share"] + shares["residual_share"] == (
        pytest.approx(1.0)
    )
    with pytest.raises(ValueError, match="constant"):
        two_way_score_shares(np.full((2, 3), 7.0))


def test_loss_model_decomposition_flip_on_planted_matrices() -> None:
    # Hand-planted score matrices: raw variation lives across losses (rows),
    # aligned variation across models (columns) — the paper's central flip.
    loss_offsets = np.array([0.55, 0.30, 0.0, -0.25, -0.50])[:, None]
    model_offsets = np.array([0.0, 0.06, 0.14, 0.25])[None, :]
    rng = np.random.default_rng(9)
    interaction = rng.normal(0.0, 0.005, (5, 4))
    raw = 0.9 + 4.0 * loss_offsets + 0.2 * model_offsets + interaction
    aligned = 0.9 + 0.02 * loss_offsets + 2.0 * model_offsets + interaction
    dec = loss_model_decomposition(raw, aligned)
    assert dec["n_blocks"] == 1
    assert dec["raw"]["ratio_median"] > 1.0
    assert dec["aligned"]["ratio_median"] < 1.0
    assert dec["flip_loss_to_model"] is True
    assert dec["raw"]["loss_share_median"] > dec["raw"]["model_share_median"]
    assert dec["aligned"]["model_share_median"] > dec["aligned"]["loss_share_median"]


def test_loss_model_decomposition_block_stack_and_fail_closed() -> None:
    stack_raw = np.stack([np.array([[0.0, 0.0], [1.0, 1.0]])] * 3)
    stack_aligned = np.stack([np.array([[0.0, 1.0], [0.0, 1.0]])] * 3)
    dec = loss_model_decomposition(stack_raw, stack_aligned)
    assert dec["n_blocks"] == 3
    assert len(dec["raw"]["ratio_by_block"]) == 3
    assert all(r == float("inf") for r in dec["raw"]["ratio_by_block"])
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_model_decomposition(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        loss_model_decomposition(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))


# ---------------------------------------------------------------------------
# VaR thresholds and breach rates — paper §4/§5.3
# ---------------------------------------------------------------------------


def test_gaussian_var_threshold_formula_and_guards() -> None:
    f = np.array([1.0, 4.0, 0.01])
    thr = gaussian_var_threshold(f, 0.05)
    np.testing.assert_allclose(thr, 1.6448536269514722 * np.sqrt(f), rtol=1e-12)
    for bad_alpha in (0.0, -0.1, 0.5, 0.95, np.nan):
        with pytest.raises(ValueError):
            gaussian_var_threshold(f, bad_alpha)
    with pytest.raises(ValueError):
        gaussian_var_threshold(np.array([1.0, -2.0]), 0.05)
    with pytest.raises(ValueError):
        gaussian_var_threshold(np.array([]), 0.05)


def test_var_breach_rate_calibrated_at_nominal() -> None:
    # True-variance forecasts under heteroskedastic normal returns breach at
    # exactly alpha in expectation; seeded, tight window.
    rng = np.random.default_rng(21)
    n = 40000
    sigma2 = np.exp(rng.normal(np.log(1e-4), 0.7, n))
    returns = np.sqrt(sigma2) * rng.normal(0.0, 1.0, n)
    res = var_breach_rates(returns, sigma2, alpha=0.05)
    assert res["n"] == n
    assert res["breach_rate"] == pytest.approx(0.05, abs=0.005)
    assert res["kupiec_p"] > 1e-3  # Kupiec does not reject correct coverage


def test_var_breach_rate_miscalibration_rejected_by_kupiec() -> None:
    rng = np.random.default_rng(22)
    n = 20000
    sigma2 = np.full(n, 1e-4)
    returns = np.sqrt(sigma2) * rng.normal(0.0, 1.0, n)
    halved = var_breach_rates(returns, 0.5 * sigma2, alpha=0.05)
    # Threshold shrinks by sqrt(0.5) → breach P(z < -1.645/sqrt(0.5)) ≈ 12%.
    assert halved["breach_rate"] > 0.05  # halved forecast ⇒ smaller VaR ⇒ more breaches
    assert halved["kupiec_p"] < 1e-3


def test_var_breach_rates_guards() -> None:
    with pytest.raises(ValueError):
        var_breach_rates(np.zeros(5), np.ones(6))
    with pytest.raises(ValueError):
        var_breach_rates(np.array([np.nan, 0.0]), np.ones(2))
    with pytest.raises(ValueError):
        var_breach_rates(np.zeros(3), np.ones(3), alpha=0.7)
    with pytest.raises(ValueError):
        var_breach_rates(np.zeros(0), np.zeros(0))


# ---------------------------------------------------------------------------
# SYNTHETIC planted world — determinism and fail-closed simulation guards
# ---------------------------------------------------------------------------


def test_world_determinism_and_seed_sensitivity() -> None:
    a = simulate_loss_model_world(600, 42)
    b = simulate_loss_model_world(600, 42)
    c = simulate_loss_model_world(600, 43)
    assert isinstance(a, LossModelWorld)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.forecasts_raw, b.forecasts_raw)
    np.testing.assert_array_equal(a.proxy, a.returns**2)
    assert not np.array_equal(a.returns, c.returns)
    assert a.forecasts_raw.shape == (4, 7, 600)
    assert a.config["data_label"] == "SYNTHETIC"


def test_world_planted_structure() -> None:
    # Forecasts factorize as c_l * base_m(t) * (small loss-specific noise):
    # the loss index moves the LEVEL only, up to the planted κ perturbation.
    w = simulate_loss_model_world(800, 7)
    ratios = w.forecasts_raw[:, 3, :] / w.forecasts_raw[:, 0, :]  # qlike / mse_log
    expected = DEFAULT_LEVEL_FACTORS[3] / DEFAULT_LEVEL_FACTORS[0]
    assert np.median(ratios) == pytest.approx(expected, rel=0.02)
    # Every forecast is strictly positive variance scale.
    assert np.all(w.forecasts_raw > 0.0)
    assert np.all(w.sigma2_true > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5, "beta": 0.6},  # persistence >= 1
        {"n": 50},
        {"level_factors": (1.0, -2.0)},
        {"tracking_weights": (0.5, 1.5)},
        {"noise_scales": (0.1,)},
        {"model_names": ("dup", "dup")},
        {"omega": -1.0},
        {"loss_movement_noise": -0.5},
        {"seed": 1.5},
    ],
)
def test_world_fail_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        simulate_loss_model_world(**{"n": 400, "seed": 3, **kwargs})


# ---------------------------------------------------------------------------
# Full study — the paper's flip and narrowing claims on SYNTHETIC data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def studies() -> dict[int, dict[str, Any]]:
    return {seed: run_loss_model_decomposition_study(seed, n=2600) for seed in STUDY_SEEDS}


def test_study_decomposition_flip_loss_to_model(studies: dict[int, dict[str, Any]]) -> None:
    for seed, res in studies.items():
        dec = res["decomposition_qlike"]
        assert res["flip_loss_to_model"] is True, seed
        # Raw: loss choice dominates (paper median 2.91); aligned: model
        # choice dominates (paper median 0.67). Wide margins vs every seed.
        assert dec["raw"]["ratio_median"] > 2.0, seed
        assert dec["aligned"]["ratio_median"] < 0.5, seed
        # Supplementary eta² split flips the same way.
        assert dec["raw"]["loss_share_median"] > dec["raw"]["model_share_median"], seed
        assert dec["aligned"]["model_share_median"] > dec["aligned"]["loss_share_median"], seed
        # Robustness score (paper: ratio stays below one under MSE-log too).
        dec_ml = res["decomposition_mse_log"]
        assert dec_ml["raw"]["ratio_median"] > 1.0, seed
        assert dec_ml["aligned"]["ratio_median"] < 1.0, seed


def test_study_breach_spread_narrows_after_alignment(
    studies: dict[int, dict[str, Any]],
) -> None:
    for seed, res in studies.items():
        assert res["breach_spread_aligned_mean"] < res["breach_spread_raw_mean"], seed
        # Paper removes 97% of cross-loss breach variation; planted world
        # reproduces the mechanism (observed > 90%).
        assert res["breach_spread_narrowing"] > 0.5, seed
        raw_by_loss = np.asarray(res["breach_rate_by_loss_raw"])
        aligned_by_loss = np.asarray(res["breach_rate_by_loss_aligned"])
        assert (raw_by_loss.max() - raw_by_loss.min()) > 5 * (
            aligned_by_loss.max() - aligned_by_loss.min()
        ), seed
        # Raw pattern mirrors paper Table 4: log-error losses breach most,
        # HMSE is the most conservative.
        assert raw_by_loss[0] > res["alpha_var"] > raw_by_loss[-1], seed


def test_study_alignment_recovers_planted_levels(studies: dict[int, dict[str, Any]]) -> None:
    factors = np.asarray(DEFAULT_LEVEL_FACTORS)
    for seed, res in studies.items():
        constants = np.asarray(res["alignment_constants"])  # (n_models, n_losses)
        rel_err = np.abs(constants * factors[None, :] - 1.0)
        assert np.max(rel_err) < 0.35, seed  # c_l ≈ 1/level_factor_l
        # LevelShare is high: planted cross-loss differences are mostly level
        # (paper: 56-89% by model; κ perturbation keeps it non-degenerate).
        assert res["level_share_mean"] > 0.6, seed
        assert min(res["level_share_by_model"]) > 0.5, seed


def test_study_alignment_transfers_honestly_to_test(
    studies: dict[int, dict[str, Any]],
) -> None:
    # Validation-fitted alignment must IMPROVE test QLIKE for mis-leveled
    # losses without ever touching test data — honest out-of-window transfer.
    for seed, res in studies.items():
        raw = np.asarray(res["scores_raw"]["qlike"])  # (n_losses, n_models)
        aligned = np.asarray(res["scores_aligned"]["qlike"])
        extreme_losses = [0, len(DEFAULT_LEVEL_FACTORS) - 1]  # c = 0.65 and 2.85
        for loss_row in extreme_losses:
            assert np.all(aligned[loss_row] < raw[loss_row]), seed


def test_study_determinism(studies: dict[int, dict[str, Any]]) -> None:
    seed = STUDY_SEEDS[0]
    rerun = run_loss_model_decomposition_study(seed, n=2600)
    digest_a = hash_bytes(canonical_json_bytes(studies[seed]))
    digest_b = hash_bytes(canonical_json_bytes(rerun))
    assert digest_a == digest_b


def test_study_honesty_contract(studies: dict[int, dict[str, Any]]) -> None:
    for res in studies.values():
        assert res["data_label"] == "SYNTHETIC"
        assert res["live_pnl_claim"] is False
        assert res["claim"] == "research_only"
        assert res["schema"] == "vol_loss_decomposition.v1"
        blob = {key: value for key, value in res.items() if key != "live_pnl_claim"}
        assert family_blob_forbidden_metrics_absent(blob)


def test_study_fail_closed_windows() -> None:
    with pytest.raises(ValueError, match="cannot host"):
        run_loss_model_decomposition_study(0, n=100, n_validation=60, n_test=60)
    with pytest.raises(ValueError):
        run_loss_model_decomposition_study(0, n=400, n_validation=10, n_test=60)
    with pytest.raises(ValueError):
        run_loss_model_decomposition_study(0, n=400, alpha_var=0.9)
    with pytest.raises(ValueError):
        run_loss_model_decomposition_study(0.5, n=400)
