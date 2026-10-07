"""C-USIM HPD split conformal (arXiv:2609.34887) -- SYNTHETIC correctness tests.

Seeded simulations only: score definition vs brute force, region geometry on a
bimodal cloud, split-conformal calibration composition, the Thm. 1 coverage
gap bound on hand-computed bin vectors, the percentile rank-score diagnostic,
fail-closed edges, and the bimodal validation bench (C-USIM region smaller
than the connected absolute-residual interval at matched marginal coverage).
Correctness material, never market evidence. No Sharpe.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.hpd_conformal import (
    HPDScorer,
    bench_cusim_bimodal,
    coverage_gap_bound,
    cusim_calibrate,
    cusim_predict,
    hpd_score,
    rank_score_percentiles,
)

SIGMA = 0.6
HI = 3.0
LO = -3.0


def _mix_density(w: float = 0.5, hi: float = HI, lo: float = LO):
    def f(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=float)
        ph = np.exp(-0.5 * ((yy - hi) / SIGMA) ** 2)
        pl = np.exp(-0.5 * ((yy - lo) / SIGMA) ** 2)
        return np.asarray((w * ph + (1.0 - w) * pl) / (SIGMA * np.sqrt(2.0 * np.pi)), float)

    return f


def _mix_cloud(rng: np.random.Generator, n: int, w: float = 0.5) -> np.ndarray:
    loc = np.where(rng.random(n) < w, HI, LO)
    return np.asarray(loc + SIGMA * rng.standard_normal(n), dtype=float)


def test_hpd_score_range_and_ordering() -> None:
    rng = np.random.default_rng(101)
    cloud = _mix_cloud(rng, 2000)
    f = _mix_density()
    ys = np.array([HI, 3.6, 1.5, 0.0, 10.0])
    s = hpd_score(cloud, ys, density=f)
    assert s.shape == (5,)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert s[0] < 0.25  # at the mode: little mass has higher density
    assert s[2] > s[0]  # shoulder above mode
    assert s[3] > 0.9  # low-density gap between modes
    assert s[4] == 1.0  # far tail: every cloud point has >= density
    grid = np.linspace(HI, 10.0, 40)
    sg = hpd_score(cloud, grid, density=f)
    assert np.all(np.diff(sg) >= 0.0)  # score grows as density falls


def test_hpd_score_matches_bruteforce_exactly() -> None:
    rng = np.random.default_rng(7)
    f = _mix_density(w=0.35)
    cloud = _mix_cloud(rng, 300, w=0.35)
    ys = rng.uniform(-6.0, 6.0, 25)
    fast = hpd_score(cloud, ys, density=f)
    d_cloud = f(cloud)
    d_y = f(ys)
    brute = np.mean(d_cloud[None, :] >= d_y[:, None], axis=1)
    assert np.array_equal(fast, brute)  # integer counts: exact, not approximate


def test_hpd_score_vectorized_matches_scalar() -> None:
    rng = np.random.default_rng(11)
    cloud = _mix_cloud(rng, 500)
    f = _mix_density()
    ys = np.array([-3.0, 0.5, 2.8, 7.0])
    vec = hpd_score(cloud, ys, density=f)
    for i, y in enumerate(ys):
        assert vec[i] == pytest.approx(float(hpd_score(cloud, y, density=f)[0]), abs=0.0)


def test_hpd_score_default_kde_deterministic() -> None:
    rng = np.random.default_rng(23)
    cloud = rng.standard_normal(1000)
    a = hpd_score(cloud, np.array([0.0, 8.0]))
    b = hpd_score(cloud, np.array([0.0, 8.0]))
    assert np.array_equal(a, b)
    assert 0.0 <= a[0] <= 0.5  # unimodal center is high-density
    assert a[1] == 1.0  # far tail below every cloud density


def test_hpd_score_fail_closed() -> None:
    rng = np.random.default_rng(3)
    cloud = _mix_cloud(rng, 100)
    with pytest.raises(ValueError, match="non-empty"):
        hpd_score(np.array([]), np.array([0.0]), density=_mix_density())
    with pytest.raises(ValueError, match="finite"):
        hpd_score(np.array([1.0, np.nan]), np.array([0.0]), density=_mix_density())
    with pytest.raises(ValueError, match="finite"):
        hpd_score(cloud, np.array([np.nan]), density=_mix_density())
    with pytest.raises(ValueError, match="distinct"):
        hpd_score(np.ones(100), np.array([1.0]))  # KDE singular on constant cloud
    with pytest.raises(ValueError, match="one value per sample"):
        HPDScorer(cloud, _mix_density(), sample_densities=np.ones(10))
    scorer = HPDScorer(cloud, None, sample_densities=np.ones(cloud.size))
    with pytest.raises(ValueError, match="density evaluator is required"):
        scorer.score(np.array([0.0]))


def test_step_density_sample_densities_shortcut() -> None:
    """Piecewise-constant (paper App. B.1 style) density, unnormalized is fine."""

    def step(y: np.ndarray) -> np.ndarray:
        a = np.abs(np.asarray(y, dtype=float))
        out = np.where(a < 1.0, 1.0, 0.2)
        return np.asarray(np.where(a >= 3.0, 0.0, out), dtype=float)

    rng = np.random.default_rng(5)
    inner = rng.uniform(-1.0, 1.0, 600)
    sign = np.where(rng.random(400) < 0.5, -1.0, 1.0)
    outer = sign * rng.uniform(1.0, 3.0, 400)
    cloud = np.concatenate([inner, outer])
    d = step(cloud)
    s_in = hpd_score(cloud, np.array([0.3]), step, sample_densities=d)
    assert s_in[0] == pytest.approx(600.0 / 1000.0, abs=0.0)  # ties share one level
    s_ann = hpd_score(cloud, np.array([1.5]), step, sample_densities=d)
    s_out = hpd_score(cloud, np.array([5.0]), step, sample_densities=d)
    assert s_ann[0] == 1.0  # every cloud density >= 0.2
    assert s_out[0] == 1.0


def test_region_bimodal_two_components_excludes_gap() -> None:
    rng = np.random.default_rng(43)
    cloud = _mix_cloud(rng, 4000)
    region = HPDScorer(cloud, _mix_density()).region(0.9)
    assert region.n_components == 2
    assert region.intervals.shape == (2, 2)
    assert region.intervals[0, 1] < region.intervals[1, 0]  # disjoint, ordered
    assert region.intervals[0, 1] < -1.0 < 1.0 < region.intervals[1, 0]  # both modes
    assert not np.any((region.members > -1.0) & (region.members < 1.0))  # gap excluded
    span = float(region.members.max() - region.members.min())
    assert region.total_length < span  # total length excludes the gap
    assert 3.0 < region.total_length < 5.0
    assert region.n_members == int(region.members.size)
    assert region.threshold == 0.9


def test_region_nested_in_threshold() -> None:
    rng = np.random.default_rng(17)
    scorer = HPDScorer(_mix_cloud(rng, 2000), _mix_density())
    prev_mask = None
    prev_len = -1.0
    for t in (0.3, 0.5, 0.7, 0.9):
        mask = scorer.region_mask(t)
        if prev_mask is not None:
            assert np.all(mask[prev_mask])  # HPD regions are nested
        reg = scorer.region(t)
        assert reg.total_length >= prev_len - 1e-12
        assert reg.n_members == int(mask.sum())
        prev_mask, prev_len = mask, reg.total_length


def test_region_empty_full_and_threshold_validation() -> None:
    rng = np.random.default_rng(19)
    scorer = HPDScorer(_mix_cloud(rng, 800), _mix_density())
    empty = scorer.region(0.0)  # every sample score >= 1/m > 0
    assert empty.n_members == 0 and empty.n_components == 0
    assert empty.intervals.shape == (0, 2) and empty.total_length == 0.0
    full = scorer.region(1.0)
    assert full.n_members == scorer.n_samples
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            scorer.region(bad)


def test_region_merge_tol_explicit_and_fail_closed() -> None:
    rng = np.random.default_rng(29)
    scorer = HPDScorer(_mix_cloud(rng, 2000), _mix_density())
    coarse = scorer.region(0.9, merge_tol=5.0)
    assert coarse.n_components == 1  # tolerance bridges the inter-mode gap
    fine = scorer.region(0.9, merge_tol=0.05)
    assert fine.n_components >= 2
    assert coarse.total_length >= fine.total_length - 1e-12
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="merge_tol"):
            scorer.region(0.9, merge_tol=bad)
    degen = HPDScorer(np.full(50, 0.5), lambda y: np.ones_like(np.asarray(y, float)))
    with pytest.raises(ValueError, match="merge_tol explicitly"):
        degen.region(1.0)  # zero spacing: default tolerance undefined


def test_cusim_calibrate_matches_conformal_quantile() -> None:
    rng = np.random.default_rng(31)
    scores = rng.random(200)
    calib = cusim_calibrate(scores, 0.1)
    assert calib.qhat == pytest.approx(conformal_quantile(scores, 0.1), abs=0.0)
    assert calib.k == 181  # ceil(201 * 0.9), paper Alg. 1 line 10
    assert calib.n_cal == 200 and calib.alpha == 0.1
    assert 0.0 <= calib.qhat <= 1.0
    assert float(np.mean(scores <= calib.qhat)) >= 0.9  # stated coverage on cal slice


def test_cusim_calibrate_fail_closed() -> None:
    scores = np.random.default_rng(37).random(50)
    for bad_alpha in (0.0, 1.0, -0.2, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            cusim_calibrate(scores, bad_alpha)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cusim_calibrate(np.append(scores, 1.5), 0.1)
    with pytest.raises(ValueError, match="finite"):
        cusim_calibrate(np.append(scores, np.nan), 0.1)
    with pytest.raises(ValueError, match="non-empty"):
        cusim_calibrate(np.array([]), 0.1)
    with pytest.raises(ValueError, match="unattainable"):
        cusim_calibrate(np.random.default_rng(2).random(5), 0.05)  # k = 6 > n = 5


def test_cusim_predict_coverage_flag_consistent_with_region() -> None:
    rng = np.random.default_rng(53)
    cloud = _mix_cloud(rng, 4000)
    f = _mix_density()
    calib = cusim_calibrate(rng.random(300), 0.1)  # qhat ~ 0.9
    pred_in = cusim_predict(cloud, calib, density=f, y=HI)
    assert pred_in.covered is True and pred_in.score is not None
    assert pred_in.score <= calib.qhat
    lo_hi = pred_in.region.intervals
    assert any(a <= HI <= b for a, b in lo_hi)
    pred_gap = cusim_predict(cloud, calib, density=f, y=0.0)
    assert pred_gap.covered is False
    assert not any(a <= 0.0 <= b for a, b in pred_gap.region.intervals)
    pred_tail = cusim_predict(cloud, calib, density=f, y=50.0)
    assert pred_tail.covered is False and pred_tail.score == 1.0
    pred_none = cusim_predict(cloud, calib, density=f)
    assert pred_none.score is None and pred_none.covered is None
    with pytest.raises(ValueError, match="finite"):
        cusim_predict(cloud, calib, density=f, y=float("nan"))


def test_gap_bound_exact_predictor_is_pure_discreteness() -> None:
    p = np.array([0.5, 0.3, 0.2])
    gb = coverage_gap_bound(p, p)  # p_hat == p_tilde: no estimation error
    assert gb.est_error_x == 0.0
    assert gb.discreteness_x == pytest.approx(0.5)
    assert gb.b_x == pytest.approx(0.5)
    assert gb.expected_b == pytest.approx(0.5)  # one-draw proxy documented
    assert gb.bound == pytest.approx(1.0)
    assert gb.n_pop == 1


def test_gap_bound_hand_example_with_population() -> None:
    p_hat = np.array([0.5, 0.3, 0.2])
    p_til = np.array([0.4, 0.4, 0.2])
    pop_hat = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    pop_til = np.array([[0.4, 0.4, 0.2], [0.5, 0.3, 0.2]])
    gb = coverage_gap_bound(p_hat, p_til, p_hat_pop=pop_hat, p_tilde_pop=pop_til)
    assert gb.est_error_x == pytest.approx(0.1)  # (1/2) * (0.1 + 0.1)
    assert gb.discreteness_x == pytest.approx(0.5)
    assert gb.b_x == pytest.approx(0.6)
    assert gb.expected_b == pytest.approx(0.65)  # rows: 0.6 and 0.7
    assert gb.bound == pytest.approx(1.25)  # Thm. 1: E_X[B(X)] + B(x)
    assert gb.n_pop == 2


def test_gap_bound_fail_closed() -> None:
    p = np.array([0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="sum to 1"):
        coverage_gap_bound(np.array([0.5, 0.3, 0.1]), p)
    with pytest.raises(ValueError, match="non-negative"):
        coverage_gap_bound(np.array([0.8, 0.4, -0.2]), p)
    with pytest.raises(ValueError, match="bin probabilities"):
        coverage_gap_bound(np.array([0.5, 0.5]), p)
    with pytest.raises(ValueError, match="finite"):
        coverage_gap_bound(np.array([0.5, np.nan, 0.5]), p)
    with pytest.raises(ValueError, match="together"):
        coverage_gap_bound(p, p, p_hat_pop=p.reshape(1, -1))
    with pytest.raises(ValueError, match="2-D"):
        coverage_gap_bound(p, p, p_hat_pop=p, p_tilde_pop=p)
    with pytest.raises(ValueError, match="sum to 1"):
        coverage_gap_bound(
            p,
            p,
            p_hat_pop=np.array([[0.5, 0.3, 0.1]]),
            p_tilde_pop=np.array([[0.5, 0.3, 0.2]]),
        )


def test_rank_score_table_homogeneous_scores() -> None:
    rng = np.random.default_rng(59)
    draws = rng.random((50, 1000))
    obs = rng.random(50)
    tbl = rank_score_percentiles(draws, 0.9, target=0.9, observed_scores=obs)
    assert tbl.conditional_coverage.shape == (50,)
    assert tbl.summary["cov_mean"] == pytest.approx(0.9, abs=0.02)
    assert tbl.summary["cov_spread_p95_p05"] < 0.08
    assert tbl.summary["ccad"] < 0.03
    assert tbl.summary["n_inputs"] == 50.0 and tbl.summary["n_draws"] == 1000.0
    assert tbl.percentile_ranks is not None
    assert np.all(tbl.percentile_ranks >= 0.0) and np.all(tbl.percentile_ranks <= 1.0)
    assert 0.3 < tbl.summary["rank_mean"] < 0.7
    # rank of the maximum draw is 1; rank of a score above every draw is 1
    top = rank_score_percentiles(draws, 0.9, observed_scores=draws.max(axis=1) + 1e-9)
    assert np.all(top.percentile_ranks == 1.0)


def test_rank_score_table_detects_heterogeneity() -> None:
    rng = np.random.default_rng(61)
    hom = rng.random((50, 800))
    het = np.empty((50, 800))
    het[:25] = rng.random((25, 800)) * 0.45  # covered fully at t = 0.9
    het[25:] = 0.5 + 0.5 * rng.random((25, 800))  # covered at 0.8
    t_hom = rank_score_percentiles(hom, 0.9, target=0.9)
    t_het = rank_score_percentiles(het, 0.9, target=0.9)
    assert t_hom.conditional_coverage.std() < 0.05
    assert t_het.summary["cov_spread_p95_p05"] > 3.0 * t_hom.summary["cov_spread_p95_p05"]
    assert t_het.summary["ccad"] > t_hom.summary["ccad"]
    lo, hi = t_het.conditional_coverage[:25], t_het.conditional_coverage[25:]
    assert np.all(lo > 0.95) and np.all(hi < 0.85)


def test_rank_score_fail_closed() -> None:
    rng = np.random.default_rng(67)
    good = rng.random((10, 50))
    with pytest.raises(ValueError, match="n_inputs, n_draws"):
        rank_score_percentiles(rng.random(50), 0.9)
    with pytest.raises(ValueError, match="finite"):
        rank_score_percentiles(np.full((2, 2), np.nan), 0.9)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rank_score_percentiles(np.full((2, 2), 1.5), 0.9)
    with pytest.raises(ValueError, match="threshold"):
        rank_score_percentiles(good, float("nan"))
    with pytest.raises(ValueError, match="target"):
        rank_score_percentiles(good, 0.9, target=1.5)
    with pytest.raises(ValueError, match="per input row"):
        rank_score_percentiles(good, 0.9, observed_scores=rng.random(9))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rank_score_percentiles(good, 0.9, observed_scores=np.full(10, -0.5))


BENCH_KEYS = {
    "synthetic_dgp",
    "synthetic_claim",
    "synthetic_seed",
    "synthetic_alpha",
    "synthetic_n_cal",
    "synthetic_n_test",
    "synthetic_n_cloud",
    "synthetic_n_diag",
    "synthetic_n_draws",
    "synthetic_qhat_cusim",
    "synthetic_qhat_miscal",
    "synthetic_qhat_absresid",
    "synthetic_coverage_cusim",
    "synthetic_coverage_absresid",
    "synthetic_mean_total_length_cusim",
    "synthetic_mean_width_absresid",
    "synthetic_length_ratio",
    "synthetic_n_components_mean",
    "synthetic_coverage_spread_wellcal",
    "synthetic_coverage_spread_miscal",
    "synthetic_coverage_std_wellcal",
    "synthetic_coverage_std_miscal",
    "synthetic_ccad_wellcal",
    "synthetic_ccad_miscal",
    "synthetic_gap_bound_wellcal_mean",
    "synthetic_gap_bound_miscal_mean",
}


def test_bench_bimodal_cusim_smaller_at_matched_coverage() -> None:
    """SYNTHETIC validation: disjoint HPD region beats the connected interval.

    The absolute-residual baseline must span the low-density gap between the
    modes (half-width qhat > 3 > |mu(x)| bounds, width > mode separation 6),
    while C-USIM keeps marginal coverage and drops the gap from the set.
    """
    row = bench_cusim_bimodal()
    assert set(row) == BENCH_KEYS
    assert row["synthetic_claim"] == "research_metric_only"
    assert row["synthetic_dgp"] == "synthetic_bimodal_mixture"
    nominal = 1.0 - float(row["synthetic_alpha"])
    for k, v in row.items():
        if k not in ("synthetic_dgp", "synthetic_claim"):
            assert np.isfinite(float(v)), k
    # matched marginal coverage: both methods valid within tolerance
    assert float(row["synthetic_coverage_cusim"]) >= nominal - 0.03
    assert float(row["synthetic_coverage_absresid"]) >= nominal - 0.03
    # C-USIM region strictly smaller; baseline interval covers the mode gap
    assert float(row["synthetic_mean_total_length_cusim"]) < float(
        row["synthetic_mean_width_absresid"]
    )
    assert float(row["synthetic_length_ratio"]) < 0.75
    assert float(row["synthetic_mean_width_absresid"]) > 6.0  # mode separation spanned
    assert float(row["synthetic_qhat_absresid"]) > 3.0  # half-width > mode offset
    # region is genuinely disconnected around the two modes
    assert float(row["synthetic_n_components_mean"]) > 1.5
    # rank-score diagnostic detects the miscalibrated predictor's heterogeneity
    assert float(row["synthetic_coverage_spread_miscal"]) > 1.5 * float(
        row["synthetic_coverage_spread_wellcal"]
    )
    assert float(row["synthetic_ccad_miscal"]) > float(row["synthetic_ccad_wellcal"])
    # Thm. 1 bound: estimation error dominates for the miscalibrated predictor
    assert float(row["synthetic_gap_bound_miscal_mean"]) > float(
        row["synthetic_gap_bound_wellcal_mean"]
    )


def test_bench_deterministic() -> None:
    a = bench_cusim_bimodal(seed=13)
    b = bench_cusim_bimodal(seed=13)
    assert a == b


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bench_cusim_bimodal(alpha=0.0)
    with pytest.raises(ValueError, match=">="):
        bench_cusim_bimodal(n_cal=1)
    with pytest.raises(ValueError, match="n_diag"):
        bench_cusim_bimodal(n_diag=5000)
    with pytest.raises(ValueError, match="unattainable"):
        bench_cusim_bimodal(n_cal=5, alpha=0.1)
