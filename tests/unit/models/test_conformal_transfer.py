"""SYNTHETIC correctness tests for Transported Conformal Calibration (TCC).

Method under test: Doula, "Conformal Calibration Transfer", ICML 2026,
arXiv:2609.10737 — transported split conformal (§3.2), TCC-KS (§3.3,
one-sided KS certificate + DKW inflation + alpha* = max(0, alpha - delta+)),
weighted-TCC (§3.4, post-transport density-ratio reweighting). Every fixture
here is SYNTHETIC (planted Gaussian shift in pair-feature space) — a
correctness test, never market evidence. Metrics are proper-score quantities
only (coverage, width, KS certificate, ESS); no Sharpe/P&L-family keys.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.conformal_transfer import (
    DEFAULT_WEIGHT_CLIP,
    WEIGHT_FLOOR,
    NearestNeighborTransport,
    _irls_logistic,
    bench_tcc,
    domain_ratio_weights,
    one_sided_ks_gap,
    synthetic_paired_shift,
    tcc_ks,
    transport_calibration,
    weighted_tcc,
)

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
# TCC-KS is conservative by construction (Thm 3.2), so it should not dip.
TOL_KS = 0.01
# weighted-TCC: ~2 standard errors of a weighted quantile under residual
# shift (the paper also reports mild stress under-coverage at severe shift,
# flagged by ESS%; see Table 2 of arXiv:2609.10737).
TOL_WEIGHTED = 0.03


def _pipeline(
    *,
    target_scale: float,
    seed: int,
    n_cal: int = 1500,
    n_pairs_fit: int = 2000,
    n_pairs_eval: int = 4000,
    n_test: int = 6000,
    eta: float = 0.1,
    alpha: float = ALPHA,
) -> SimpleNamespace:
    """Run the full TCC pipeline on the synthetic fixture (paper Fig. 2)."""
    data = synthetic_paired_shift(
        n_cal,
        n_pairs_fit,
        n_pairs_eval,
        n_test,
        target_scale=target_scale,
        seed=seed,
    )
    model = data.model
    transported = transport_calibration(
        data.x_src_cal,
        data.y_cal,
        data.x_src_pairs_fit,
        data.x_tgt_pairs_fit,
        model.score,
    )
    x_eval_transported = transported.map.predict(data.x_src_pairs_eval)
    ks = tcc_ks(
        transported.scores,
        model.surrogate(x_eval_transported),
        model.surrogate(data.x_tgt_pairs_eval),
        alpha,
        eta=eta,
    )
    wt = weighted_tcc(
        transported.scores,
        model.features(transported.x_transported),
        model.features(x_eval_transported),
        model.features(data.x_tgt_pairs_eval),
        alpha,
    )

    def coverage(q: float) -> float:
        return float(np.mean(model.score(data.x_tgt_test, data.y_test) <= q))

    q_plain = conformal_quantile(transported.scores, alpha)
    return SimpleNamespace(
        data=data,
        model=model,
        transported=transported,
        ks=ks,
        wt=wt,
        q_plain=q_plain,
        cov_plain=coverage(q_plain),
        cov_ks=coverage(ks.qhat),
        cov_w=coverage(wt.qhat),
    )


def test_nearest_neighbor_transport_recovers_translation() -> None:
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(800, 2))
    offset = np.array([3.0, -1.5])
    xt = xs + offset
    tmap = NearestNeighborTransport.fit(xs, xt)
    query = rng.normal(size=(300, 2))
    pred = tmap.predict(query)
    err = np.linalg.norm(pred - (query + offset), axis=1)
    assert float(np.mean(err)) < 0.15
    # deterministic: refit + repredict is bitwise identical
    pred2 = NearestNeighborTransport.fit(xs, xt).predict(query)
    assert np.array_equal(pred, pred2)


def test_transport_fit_and_predict_fail_closed() -> None:
    xs = np.zeros((5, 2))
    xt = np.ones((5, 2))
    with pytest.raises(ValueError):
        NearestNeighborTransport.fit(xs, xt, k_neighbors=6)  # k > m
    with pytest.raises(ValueError):
        NearestNeighborTransport.fit(xs, xt, k_neighbors=0)
    with pytest.raises(ValueError):
        NearestNeighborTransport.fit(xs, np.ones((4, 2)))  # pair shape mismatch
    tmap = NearestNeighborTransport.fit(xs, xt, k_neighbors=3)
    with pytest.raises(ValueError):
        tmap.predict(np.zeros((2, 3)))  # feature-dim mismatch
    with pytest.raises(ValueError):
        tmap.predict(np.full((2, 2), np.nan))  # non-finite


def test_transport_calibration_fail_closed() -> None:
    data = synthetic_paired_shift(n_cal=40, n_pairs_fit=60, n_pairs_eval=10, n_test=10, seed=0)
    model = data.model
    with pytest.raises(ValueError):
        transport_calibration(
            data.x_src_cal[:39],
            data.y_cal,
            data.x_src_pairs_fit,
            data.x_tgt_pairs_fit,
            model.score,
        )  # label/row mismatch
    with pytest.raises(ValueError):
        transport_calibration(
            data.x_src_cal,
            data.y_cal,
            data.x_src_pairs_fit,
            data.x_tgt_pairs_fit[:59],
            model.score,
        )  # pair shape mismatch
    bad = data.x_src_cal.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        transport_calibration(
            bad, data.y_cal, data.x_src_pairs_fit, data.x_tgt_pairs_fit, model.score
        )  # non-finite inputs
    with pytest.raises(ValueError):
        transport_calibration(
            data.x_src_cal,
            data.y_cal,
            data.x_src_pairs_fit,
            data.x_tgt_pairs_fit,
            lambda x, y: np.zeros(len(y) - 1),
        )  # score_fn must return one score per row
    with pytest.raises(ValueError):
        transport_calibration(
            data.x_src_cal,
            data.y_cal,
            data.x_src_pairs_fit,
            data.x_tgt_pairs_fit,
            model.score,
            k_neighbors=61,
        )  # k > number of fit pairs


def test_one_sided_ks_gap_hand_cases() -> None:
    lo = np.zeros(50)
    hi = np.ones(50)
    # transported surrogate strictly below target -> maximal "target harder" gap
    assert one_sided_ks_gap(lo, hi) == pytest.approx(1.0)
    # opposite direction is conservative and needs no correction -> 0
    assert one_sided_ks_gap(hi, lo) == 0.0
    assert one_sided_ks_gap(lo, lo.copy()) == 0.0
    half = np.concatenate([np.zeros(25), np.ones(25)])
    assert one_sided_ks_gap(half, hi) == pytest.approx(0.5)


def test_tcc_ks_certificate_identity_and_max_convention() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=200) + 3.0
    surr_lo = np.zeros(1000)
    surr_hi = np.ones(1000)
    dkw = 2.0 * np.sqrt(np.log(4.0 / 0.1) / (2.0 * 1000.0))
    # extreme certified mismatch: alpha* clips to 0 and qhat is S_(n)
    # (paper footnote 1: sample max, never +inf)
    res = tcc_ks(scores, surr_lo, surr_hi, alpha=ALPHA, eta=0.1)
    assert res.delta_hat == pytest.approx(1.0)
    assert res.delta_plus == pytest.approx(1.0 + dkw)
    assert res.alpha_star == 0.0
    assert res.qhat == pytest.approx(float(np.max(scores)))
    # no mismatch: alpha* = alpha - DKW floor, threshold is the alpha*
    # order statistic and strictly below the vacuous max
    res0 = tcc_ks(scores, surr_lo, surr_lo.copy(), alpha=ALPHA, eta=0.1)
    assert res0.delta_hat == 0.0
    assert res0.alpha_star == pytest.approx(ALPHA - dkw)
    assert 0.0 < res0.alpha_star < ALPHA
    assert res0.qhat == pytest.approx(conformal_quantile(scores, res0.alpha_star))
    assert res0.qhat < float(np.max(scores))


def test_tcc_ks_fail_closed() -> None:
    scores = np.arange(10, dtype=float)
    surr = np.arange(10, dtype=float)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            tcc_ks(scores, surr, surr, alpha=bad_alpha)
    for bad_eta in (0.0, 1.0):
        with pytest.raises(ValueError):
            tcc_ks(scores, surr, surr, alpha=ALPHA, eta=bad_eta)
    with pytest.raises(ValueError):
        tcc_ks(np.array([np.nan, 1.0]), surr, surr, alpha=ALPHA)
    with pytest.raises(ValueError):
        tcc_ks(scores, np.array([]), surr, alpha=ALPHA)


def test_domain_ratio_weights_bounds_and_separation() -> None:
    rng = np.random.default_rng(2)
    n = 500
    pool0 = rng.normal(size=(n, 2))
    pool1 = rng.normal(size=(n, 2)) + np.array([4.0, 0.0])
    w = domain_ratio_weights(pool0, pool0, pool1, weight_clip=DEFAULT_WEIGHT_CLIP)
    assert w.shape == (n,)
    assert np.all(w >= WEIGHT_FLOOR)
    assert np.all(w <= DEFAULT_WEIGHT_CLIP + 1e-12)
    # rows that look transported get odds far below 1 (tail overlap excepted)
    assert float(np.median(w)) < 0.1
    assert float(np.mean(w < 1.0)) > 0.9
    # rows that look like the real target saturate the clip under separation
    w_target_like = domain_ratio_weights(pool1, pool0, pool1, weight_clip=DEFAULT_WEIGHT_CLIP)
    assert float(np.mean(w_target_like >= DEFAULT_WEIGHT_CLIP - 1e-12)) > 0.5
    with pytest.raises(ValueError):
        domain_ratio_weights(pool0, pool0, pool1, weight_clip=WEIGHT_FLOOR)
    with pytest.raises(ValueError):
        domain_ratio_weights(pool0, pool0, rng.normal(size=(n, 3)))


def test_weighted_tcc_reduces_to_uniform_without_shift() -> None:
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(size=800)) + 0.5
    feats = rng.normal(size=(800, 2))
    pool_a = rng.normal(size=(600, 2))
    pool_b = rng.normal(size=(600, 2))
    res = weighted_tcc(scores, feats, pool_a, pool_b, ALPHA)
    # identical pools -> weights ~ 1, ESS high, threshold ~ the plain
    # split-conformal order statistic (Tibshirani uniform-weight identity)
    assert np.all(np.abs(res.weights - 1.0) < 0.5)
    assert res.ess_percent > 80.0
    assert res.qhat == pytest.approx(conformal_quantile(scores, ALPHA), rel=0.08)


def test_weighted_tcc_fail_closed() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(size=50) + 3.0
    feats = rng.normal(size=(50, 2))
    pool = rng.normal(size=(60, 2))
    with pytest.raises(ValueError):
        weighted_tcc(scores, feats[:49], pool, pool, ALPHA)  # row mismatch
    with pytest.raises(ValueError):
        weighted_tcc(scores, feats, pool, pool, 0.0)  # alpha
    with pytest.raises(ValueError):
        weighted_tcc(scores, feats, pool, pool, ALPHA, weight_clip=WEIGHT_FLOOR)
    with pytest.raises(ValueError):
        weighted_tcc(scores, feats, pool, rng.normal(size=(60, 3)), ALPHA)


def test_synthetic_generator_fail_closed() -> None:
    with pytest.raises(ValueError):
        synthetic_paired_shift(n_cal=0)
    with pytest.raises(ValueError):
        synthetic_paired_shift(target_scale=0.0)
    with pytest.raises(ValueError):
        synthetic_paired_shift(kappa=-1.0)
    with pytest.raises(ValueError):
        synthetic_paired_shift(dim=0)
    with pytest.raises(ValueError):
        synthetic_paired_shift(shift=float("nan"))


def test_synthetic_mismatch_transport_only_undercovers_tcc_fixes() -> None:
    """The gap TCC fixes: plain transported CP undercovers under mismatch.

    Planted Gaussian shift in pair-feature space (target_scale=1.4): the
    transported calibration misses target-domain coverage, while both TCC
    variants restore it (TCC-KS conservatively per Thm 3.2, weighted-TCC
    within weighted-quantile noise per Prop. 3.3).
    """
    run = _pipeline(target_scale=1.4, seed=7)
    # plain transported split conformal UNDERCOVERS the shifted target domain
    assert run.cov_plain <= NOMINAL - 0.05
    # TCC-KS: certified mismatch exceeds alpha -> alpha* = 0 and the
    # footnote-1 S_(n) convention; coverage restored with margin
    assert run.ks.delta_plus > 0.2
    assert run.ks.alpha_star == 0.0
    assert run.ks.qhat == pytest.approx(float(np.max(run.transported.scores)))
    assert run.ks.qhat > run.q_plain
    assert run.cov_ks >= NOMINAL - TOL_KS
    # weighted-TCC: reweighting toward the deployment pool restores coverage
    # within tolerance and clearly beats the uncorrected threshold
    assert run.wt.qhat > run.q_plain
    assert run.cov_w >= NOMINAL - TOL_WEIGHTED
    assert run.cov_w > run.cov_plain + 0.04
    # label-free weight-stability diagnostic is in its documented regime
    # (paper Sec. 4: moderate mismatch -> moderate ESS%)
    assert 30.0 < run.wt.ess_percent < 85.0


def test_tcc_ks_interior_regime_adjusts_level_not_vacuous() -> None:
    """Small mismatch: alpha* stays strictly inside (0, alpha) and coverage holds."""
    run = _pipeline(target_scale=1.05, seed=11, n_pairs_eval=20000)
    assert 0.0 < run.ks.alpha_star < ALPHA
    assert run.ks.qhat == pytest.approx(
        conformal_quantile(run.transported.scores, run.ks.alpha_star)
    )
    assert run.q_plain < run.ks.qhat < float(np.max(run.transported.scores))
    assert run.cov_ks >= NOMINAL - TOL_KS


def test_zero_mismatch_degrades_to_standard_conformal() -> None:
    """mismatch -> 0: TCC-KS collapses to the DKW floor and weighted-TCC to
    uniform weights, i.e. both reduce to standard (weighted) split conformal."""
    run = _pipeline(target_scale=1.0, seed=7)
    # certificate collapses toward the finite-sample DKW floor
    assert run.ks.delta_hat < 0.04
    expected_floor = 2.0 * np.sqrt(np.log(4.0 / 0.1) / (2.0 * 4000.0))
    assert run.ks.delta_plus == pytest.approx(run.ks.delta_hat + expected_floor)
    assert ALPHA - 0.07 <= run.ks.alpha_star < ALPHA
    # TCC-KS is exactly the alpha* order statistic and NOT vacuous
    assert run.ks.qhat == pytest.approx(
        conformal_quantile(run.transported.scores, run.ks.alpha_star)
    )
    assert run.ks.qhat < float(np.max(run.transported.scores))
    # weighted-TCC ~ standard weighted conformal with near-uniform weights
    assert run.wt.ess_percent >= 90.0
    assert np.all(np.abs(run.wt.weights - 1.0) < 0.5)
    assert run.wt.qhat == pytest.approx(run.q_plain, rel=0.12)
    # coverage is near nominal for all three (no mismatch to fix)
    assert abs(run.cov_plain - NOMINAL) <= 0.03
    assert NOMINAL - TOL_KS <= run.cov_ks <= NOMINAL + 0.09
    assert NOMINAL - TOL_WEIGHTED <= run.cov_w <= NOMINAL + 0.06


def test_mismatch_certificate_tracks_coverage_gap() -> None:
    """Observable delta+ must correlate with the actual (eval-only) gap."""
    scales = (1.0, 1.2, 1.4, 1.6, 1.8)
    deltas, gaps, covs_ks, covs_w, ess = [], [], [], [], []
    for scale in scales:
        run = _pipeline(target_scale=scale, seed=7)
        deltas.append(run.ks.delta_plus)
        gaps.append(NOMINAL - run.cov_plain)
        covs_ks.append(run.cov_ks)
        covs_w.append(run.cov_w)
        ess.append(run.wt.ess_percent)
    # label-free certificate is strictly monotone in the planted mismatch
    assert np.all(np.diff(deltas) > 0.0)
    # and so is the true coverage gap of uncorrected transported CP
    assert np.all(np.diff(gaps) > 0.0)
    # certificate correlates with the gap it is supposed to predict
    assert float(spearmanr(deltas, gaps).statistic) >= 0.9
    assert float(pearsonr(deltas, gaps).statistic) >= 0.7
    # TCC-KS holds nominal coverage across the whole sweep (guardrail role)
    assert min(covs_ks) >= NOMINAL - TOL_KS
    # weighted-TCC holds the tolerance in the stable-weight regime and
    # degrades gracefully under severe shift, with ESS% flagging the regime
    # (paper Tables 2-3: mild under-coverage at severe shift is expected)
    assert min(covs_w[:3]) >= NOMINAL - TOL_WEIGHTED
    assert min(covs_w) >= NOMINAL - 0.07
    assert np.all(np.diff(ess) < 0.0)


def test_surrogate_tracks_score_alignment_a2() -> None:
    """Paper A2 / rho(T,S) audit: the label-free surrogate ranks true scores.

    The paper measures rho(T,S) ~ 0.31-0.33 on CIFAR-100-C / Tiny-ImageNet-C
    (Table 5 of arXiv:2609.10737); the multiplicative label noise attenuates
    the rank correlation, so the bar is "clearly positive", not "strong".
    """
    run = _pipeline(target_scale=1.4, seed=7)
    surr = run.model.surrogate(run.data.x_tgt_test)
    s_test = run.model.score(run.data.x_tgt_test, run.data.y_test)
    rho = float(spearmanr(surr, s_test).statistic)
    assert 0.25 < rho < 0.75


def test_irls_nonconvergence_fails_closed() -> None:
    """Exhausting max_iter without meeting tol must raise — silently
    returning the un-converged iterate fakes a fitted domain classifier."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    labels = (x[:, 0] > 0.0).astype(float)
    with pytest.raises(ValueError, match="converge"):
        _irls_logistic(x, labels, l2=1e-2, max_iter=1)


def test_bench_tcc_keys_proper_scores_only() -> None:
    row = bench_tcc(n_cal=800, n_pairs_fit=1000, n_pairs_eval=2000, n_test=3000, seed=11)
    for key in (
        "synthetic_coverage_transport_only",
        "synthetic_coverage_tcc_ks",
        "synthetic_coverage_weighted_tcc",
        "synthetic_mean_width_transport_only",
        "synthetic_mean_width_tcc_ks",
        "synthetic_mean_width_weighted_tcc",
        "synthetic_qhat_transport_only",
        "synthetic_qhat_tcc_ks",
        "synthetic_qhat_weighted_tcc",
        "synthetic_delta_hat",
        "synthetic_delta_plus",
        "synthetic_alpha_star",
        "synthetic_ess_percent",
        "synthetic_n",
        "synthetic_alpha",
        "synthetic_target_scale",
        "synthetic_seed",
    ):
        assert key in row
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in row:
        assert "sharpe" not in key.lower()
        assert not (set(key.lower().split("_")) & forbidden)
    assert row["synthetic_dgp"] == "fixture"
    assert row["synthetic_claim"] == "research_metric_only"
    assert float(row["synthetic_n"]) > 0.0
    # the corrected variants are at least as wide/covering as plain transport
    assert float(row["synthetic_coverage_tcc_ks"]) >= float(
        row["synthetic_coverage_transport_only"]
    )
    assert float(row["synthetic_mean_width_tcc_ks"]) >= float(
        row["synthetic_mean_width_transport_only"]
    )


def test_bench_tcc_deterministic() -> None:
    kwargs = {
        "n_cal": 600,
        "n_pairs_fit": 800,
        "n_pairs_eval": 1000,
        "n_test": 2000,
        "seed": 3,
    }
    assert bench_tcc(**kwargs) == bench_tcc(**kwargs)
