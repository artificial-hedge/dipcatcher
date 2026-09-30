"""Tests for quant_fund.models.rolling_conformal — rolling-CP (Cheng, Liang &
Barber 2026, arXiv:2609.26951). SYNTHETIC streams only: correctness tests,
never market evidence (AGENTS.md honesty contract #2). Diagnostics are
coverage / width / p-values only (contract #1).

Regime map (each Monte-Carlo test documents which regime it is in):

- STABLE (paper's Theorem 4 / Assumption 1 regime): OnlineRidge,
  OnlineMeanTracker, OnlineQuantileTracker on i.i.d. streams — score
  functions stabilize, empirical coverage approaches the nominal 1 - alpha.
- ADVERSARIAL (paper's Proposition 1, App. A.4 — worst case ATTAINED):
  AlternatingFoldScorer on i.i.d. Unif(0,1) labels — coverage converges to
  1 - 2*alpha + nu, strictly below nominal, while the Theorem 1 floor
  1 - 2*alpha holds. Honesty: this is the worst case, not typical.
- UNSTABLE-not-worst-case: LastPointOverfitter (memorizes the last label with
  growing gain) — rolling-CP holds its floor (in fact overcovers here), while
  the naive arrival-time split (calibrate at arrival, test with the final
  model) violates split-CP's fixed-score premise and collapses below any
  floor.

Determinism: every MC test pins np.random.default_rng seeds (stream seeds
``seed0 + k``, independent holdout seeds offset by >= 500_000). Tolerances are
documented per test as multiples of the Monte-Carlo standard error (SE);
assertions are reproducible bit-for-bit given the pinned seeds.
"""

import numpy as np
import pytest

from quant_fund.models.rolling_conformal import (
    AlternatingFoldScorer,
    LastPointOverfitter,
    OnlineMeanTracker,
    OnlineQuantileTracker,
    OnlineRidge,
    RollingConformal,
    SplitConformal,
    marginal_coverage_floor,
    rolling_pvalue,
    split_pvalue,
    stability_coverage_floor,
    training_conditional_miscoverage_bound,
)

ALPHA = 0.1
X0 = np.zeros(1)  # dummy feature for label-only scorers


# ---------------------------------------------------------------------------
# pure p-value functions (paper Eq. (3)/(4) and Sec. 2.2 formulas)
# ---------------------------------------------------------------------------


def test_rolling_pvalue_matches_paper_formula() -> None:
    # p_rolling = (1 + #{i : S_i >= s_i(z)}) / (n + 1)
    S = np.array([1.0, 2.0, 3.0])
    cand = np.array([0.5, 2.5, 1.0])  # S >= cand at i = 1, 3 -> count 2
    assert rolling_pvalue(S, cand) == pytest.approx((1.0 + 2.0) / 4.0)
    assert rolling_pvalue(np.empty(0), np.empty(0)) == 1.0  # C_0 = Z
    with pytest.raises(ValueError):
        rolling_pvalue(S, np.array([1.0, 2.0]))  # length mismatch
    with pytest.raises(ValueError):
        rolling_pvalue(S, np.array([0.5, np.nan, 1.0]))


def test_split_pvalue_matches_paper_formula() -> None:
    # p_split = (1 + #{i : s(z) <= S_i}) / (n1 + 1), ONE fixed score function
    S = np.array([1.0, 2.0, 3.0])
    assert split_pvalue(S, 1.5) == pytest.approx(3.0 / 4.0)
    assert split_pvalue(S, 0.5) == 1.0
    assert split_pvalue(np.empty(0), 1.5) == 1.0  # no calibration -> all of R
    with pytest.raises(ValueError):
        split_pvalue(S, np.inf)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_pvalue_covered_equivalence_with_equation_4(alpha: float) -> None:
    """Eq. (4) count form and the Sec. 2.2 p-value form are equivalent."""
    rng = np.random.default_rng(11)
    n = 37
    S = rng.normal(size=n)
    cand = rng.normal(size=n)
    p = rolling_pvalue(S, cand)
    count_strict = int(np.count_nonzero(cand > S))
    assert (p > alpha) == (count_strict < (1.0 - alpha) * (n + 1))


# ---------------------------------------------------------------------------
# driver mechanics: calibrate-then-roll, replay, intervals, validation
# ---------------------------------------------------------------------------


def test_calibrate_then_roll_order() -> None:
    """update() scores with the state trained on Z_<i, THEN rolls Z_i in."""
    rc = RollingConformal(lambda: OnlineMeanTracker(prior_mean=0.0), alpha=ALPHA)
    assert rc.update(X0, 5.0) == pytest.approx(5.0)  # |5 - prior 0|
    assert rc.update(X0, 7.0) == pytest.approx(2.0)  # |7 - mean(5)|
    assert rc.update(X0, 9.0) == pytest.approx(3.0)  # |9 - mean(5, 7)|
    assert rc.n_obs == 3
    np.testing.assert_allclose(rc.scores, [5.0, 2.0, 3.0])


def test_pvalue_matches_independent_equation_4_replay() -> None:
    """Hand-rolled Eq. (4) replay (numpy only) vs the driver, mean tracker."""
    rng = np.random.default_rng(3)
    n = 25
    y = rng.normal(size=n)
    rc = RollingConformal(lambda: OnlineMeanTracker(prior_mean=0.0), alpha=ALPHA)
    rc.fit_stream(np.zeros((n, 1)), y)
    mu = np.concatenate([[0.0], np.cumsum(y)[:-1] / np.arange(1, n)])  # mu_i, i = 1..n
    S = np.abs(y - mu)
    np.testing.assert_allclose(rc.scores, S, atol=1e-12)
    for y_star in (-1.5, -0.2, 0.0, 0.7, 2.5):
        s = np.abs(y_star - mu)
        count_ge = int(np.count_nonzero(s <= S))
        p = (1.0 + count_ge) / (n + 1.0)
        assert rc.pvalue(X0, y_star) == pytest.approx(p)
        count_strict = int(np.count_nonzero(s > S))
        assert rc.covered(X0, y_star) == (count_strict < (1.0 - ALPHA) * (n + 1))


class _ReplayOnly:
    """Hides score_trajectory so the driver must fall back to factory replay."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def fit_one(self, x: np.ndarray, y: float) -> None:
        self._inner.fit_one(x, y)  # type: ignore[attr-defined]

    def score(self, x: np.ndarray, y: float) -> float:
        return float(self._inner.score(x, y))  # type: ignore[attr-defined]


def test_replay_fallback_matches_trajectory_path() -> None:
    rng = np.random.default_rng(5)
    n = 30
    y = rng.normal(size=n)
    X = np.zeros((n, 1))
    fast = RollingConformal(lambda: OnlineMeanTracker(), alpha=ALPHA).fit_stream(X, y)
    slow = RollingConformal(lambda: _ReplayOnly(OnlineMeanTracker()), alpha=ALPHA)
    slow.fit_stream(X, y)
    for y_star in (-2.0, 0.1, 1.7):
        assert slow.pvalue(X0, y_star) == pytest.approx(fast.pvalue(X0, y_star))
        assert slow.covered(X0, y_star) == fast.covered(X0, y_star)


def test_predict_interval_matches_pvalue_membership() -> None:
    """In this (connected) regime: y in [lo, hi] iff p_rolling(y) > alpha."""
    rng = np.random.default_rng(7)
    n = 60
    y = rng.normal(size=n)
    rc = RollingConformal(lambda: OnlineMeanTracker(), alpha=ALPHA)
    rc.fit_stream(np.zeros((n, 1)), y)
    lo, hi, point = rc.predict_interval(X0)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    assert lo <= point <= hi
    grid = np.arange(-4.0, 4.001, 0.05)
    for g in grid:
        in_interval = lo <= g <= hi
        assert in_interval == rc.covered(X0, float(g)), f"mismatch at y={g:.3f}"
    # deterministic on repeat
    assert rc.predict_interval(X0) == (lo, hi, point)


def test_predict_interval_zero_width_degenerate() -> None:
    """All labels equal the (constant) forecast: zero-radius scores -> [0, 0]."""
    rc = RollingConformal(lambda: OnlineMeanTracker(prior_mean=0.0), alpha=ALPHA)
    rc.fit_stream(np.zeros((20, 1)), np.zeros(20))
    lo, hi, point = rc.predict_interval(X0)
    assert (lo, hi, point) == (0.0, 0.0, 0.0)
    assert rc.covered(X0, 0.0)  # p_rolling = 1
    assert not rc.covered(X0, 1.0)  # p_rolling = 1/21 <= alpha


class _SpikeScorer:
    """Centers march 1, 2, 3, ...; labels sit exactly on centers.

    All scores are zero (zero-radius, pairwise disjoint sublevel points), so
    no y reaches depth k_min >= 2 and the prediction set is EMPTY.
    """

    def __init__(self) -> None:
        self._n = 0

    def fit_one(self, x: np.ndarray, y: float) -> None:
        self._n += 1

    def score(self, x: np.ndarray, y: float) -> float:
        return abs(float(y) - float(self._n + 1))

    def predict(self, x: np.ndarray) -> float:
        return float(self._n + 1)

    def predict_trajectory(self, x: np.ndarray) -> np.ndarray:
        return np.arange(1, self._n + 1, dtype=float)

    def score_radius(self, s: float) -> tuple[float, float]:
        return float(s), float(s)


def test_predict_interval_empty_set_sentinel() -> None:
    rc = RollingConformal(_SpikeScorer, alpha=ALPHA)
    rc.fit_stream(np.zeros((20, 1)), np.arange(1.0, 21.0))
    lo, hi, _ = rc.predict_interval(X0)
    assert lo == np.inf and hi == -np.inf  # empty set: lo > hi
    assert not np.any(rc.evaluate(np.zeros((3, 1)), np.array([1.0, 5.0, 20.0]))[1])


def test_driver_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        RollingConformal(lambda: OnlineMeanTracker(), alpha=0.0)
    with pytest.raises(ValueError):
        RollingConformal(lambda: OnlineMeanTracker(), alpha=1.0)
    with pytest.raises(ValueError):
        RollingConformal(lambda: OnlineMeanTracker(), alpha=float("nan"))
    with pytest.raises(TypeError):
        RollingConformal(lambda: object(), alpha=ALPHA)  # not a scorer
    with pytest.raises(TypeError):
        RollingConformal(42, alpha=ALPHA)  # factory not callable

    rc = RollingConformal(lambda: OnlineMeanTracker(), alpha=ALPHA)
    with pytest.raises(ValueError):
        rc.update(np.array([np.nan]), 1.0)
    with pytest.raises(ValueError):
        rc.update(X0, float("inf"))
    with pytest.raises(ValueError):
        rc.update(np.zeros((2, 2)), 1.0)  # x must be 1-D
    rc.update(np.zeros(2), 1.0)
    with pytest.raises(ValueError):
        rc.update(np.zeros(3), 1.0)  # feature dimension changed
    with pytest.raises(ValueError):
        rc.fit_stream(np.zeros((3, 2)), np.zeros(4))  # length mismatch
    with pytest.raises(ValueError):
        rc.evaluate(np.zeros((0, 1)), np.zeros(0))  # empty test batch

    # interval construction requires alpha * (n + 1) > 1, else the set is R
    small = RollingConformal(lambda: OnlineMeanTracker(), alpha=0.05)
    small.fit_stream(np.zeros((5, 1)), np.arange(5.0))
    with pytest.raises(ValueError):
        small.predict_interval(X0)

    # interval construction requires the IntervalScorer protocol
    fold = RollingConformal(lambda: AlternatingFoldScorer(0.85, 1.0), alpha=ALPHA)
    fold.fit_stream(np.zeros((30, 1)), np.linspace(0.01, 0.99, 30))
    with pytest.raises(TypeError):
        fold.predict_interval(X0)


def test_run_stream_result_contract() -> None:
    rng = np.random.default_rng(13)
    n, d, h = 120, 3, 12
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, -0.5, 0.25]) + rng.normal(0.0, 0.4, n)
    Xt = rng.normal(size=(h, d))
    yt = Xt @ np.array([1.0, -0.5, 0.25]) + rng.normal(0.0, 0.4, h)
    res = RollingConformal(lambda: OnlineRidge(d, lam=0.2), alpha=ALPHA).run_stream(X, y, Xt, yt)
    assert res.lower.shape == res.upper.shape == res.point.shape == (h,)
    assert res.pvalues.shape == (h,) and res.covered.shape == (h,)
    assert np.all(res.pvalues > 0.0) and np.all(res.pvalues <= 1.0)
    assert np.all(res.lower <= res.upper)
    assert np.all((res.lower <= res.point) & (res.point <= res.upper))
    assert np.all(res.covered == (res.pvalues > ALPHA))
    assert 0.0 <= res.coverage <= 1.0
    assert res.mean_width >= 0.0
    assert res.coverage == pytest.approx(float(np.mean(res.covered)))


# ---------------------------------------------------------------------------
# concrete sequential predictors
# ---------------------------------------------------------------------------


def test_online_ridge_matches_batch_solve() -> None:
    """Recursive Sherman-Morrison update == batch normal-equation solve."""
    rng = np.random.default_rng(17)
    n, d, lam = 80, 5, 0.3
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    model = OnlineRidge(d, lam=lam, score="sq")
    for j in range(n):
        model.fit_one(X[j], y[j])
    theta_batch = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    np.testing.assert_allclose(model.theta, theta_batch, atol=1e-8)
    # trajectory rows are batch refits on prefixes: theta_k on X[:k], y[:k]
    traj = model.predict_trajectory(X[0])
    for k in (1, 5, 20, 60):
        th_k = np.linalg.solve(X[:k].T @ X[:k] + lam * np.eye(d), X[:k].T @ y[:k])
        assert traj[k] == pytest.approx(float(th_k @ X[0]), abs=1e-8)


def test_online_ridge_scores_radii_and_validation() -> None:
    model = OnlineRidge(2, lam=1.0, score="sq")
    model.fit_one(np.array([1.0, 0.0]), 3.0)
    # theta_1 = x y / (|x|^2 + lam) = [1.5, 0]; mu = 1.5 at x=[1,0]
    assert model.predict(np.array([1.0, 0.0])) == pytest.approx(1.5)
    assert model.score(np.array([1.0, 0.0]), 3.0) == pytest.approx(0.5 * 1.5**2)
    np.testing.assert_allclose(
        model.score_trajectory(np.array([1.0, 0.0]), 3.0),
        [0.5 * 9.0],  # mu_1 = 0
    )
    assert model.score_radius(2.0) == (2.0, 2.0)  # sqrt(2 * 2)
    assert OnlineRidge(2, score="abs").score_radius(2.0) == (2.0, 2.0)
    with pytest.raises(ValueError):
        model.score_radius(-1.0)
    with pytest.raises(ValueError):
        OnlineRidge(2, lam=0.0)  # singular recursion before the threshold
    with pytest.raises(ValueError):
        OnlineRidge(0)
    with pytest.raises(ValueError):
        OnlineRidge(2, score="huber")
    with pytest.raises(ValueError):
        model.fit_one(np.zeros(3), 1.0)  # wrong feature count


def test_quantile_tracker_matches_numpy_quantile() -> None:
    model = OnlineQuantileTracker(tau=0.5, prior=0.0)
    for yv in (1.0, 3.0, 2.0):
        model.fit_one(X0, yv)
    assert model.predict(X0) == pytest.approx(float(np.quantile([1.0, 3.0, 2.0], 0.5)))
    np.testing.assert_allclose(model.predict_trajectory(X0), [0.0, 1.0, 2.0])
    # pinball at tau: (y - q) * tau above q, (q - y) * (1 - tau) below
    assert model.score(X0, 4.0) == pytest.approx(0.5 * 2.0)
    assert model.score(X0, 0.0) == pytest.approx(0.5 * 2.0)
    # asymmetric sublevel sets [q - s/(1-tau), q + s/tau]
    t9 = OnlineQuantileTracker(tau=0.9)
    assert t9.score_radius(0.9) == (pytest.approx(9.0), pytest.approx(1.0))
    with pytest.raises(ValueError):
        OnlineQuantileTracker(tau=0.0)
    with pytest.raises(ValueError):
        OnlineQuantileTracker(tau=1.0)


def test_fold_scorer_matches_proposition_1_construction() -> None:
    """s_i(z) = z (odd i); fold z -> lo + hi - z on [lo, hi] (even i)."""
    lo, hi = 0.85, 1.0  # paper: l = 1 - 2*alpha + nu, r = 1
    sc = AlternatingFoldScorer(lo, hi)
    assert sc.score(X0, 0.5) == 0.5  # i = 1 odd, outside window: identity
    sc.fit_one(X0, 0.1)
    assert sc.score(X0, 0.9) == pytest.approx(lo + hi - 0.9)  # i = 2 even, inside
    assert sc.score(X0, 0.5) == 0.5  # i = 2 even, outside window
    sc.fit_one(X0, 0.2)
    sc.fit_one(X0, 0.3)
    np.testing.assert_allclose(sc.score_trajectory(X0, 0.9), [0.9, 0.95, 0.9])
    with pytest.raises(ValueError):
        AlternatingFoldScorer(1.0, 0.85)  # need lo < hi (lo < 1 - alpha < hi)


def test_overfitter_trajectory_and_gains() -> None:
    """mu_i = g_i * y_{i-1} with mu_1 = 0; g_i = 1 + (i-1)/gain_scale."""
    model = LastPointOverfitter(gain_scale=10.0, growth="linear")
    for yv in (2.0, -1.0, 0.5):
        model.fit_one(X0, yv)
    np.testing.assert_allclose(model.predict_trajectory(X0), [0.0, 1.1 * 2.0, 1.2 * -1.0])
    assert model.predict(X0) == pytest.approx(1.3 * 0.5)  # g_4 * y_3
    assert model.score(X0, 0.0) == pytest.approx(0.65)
    exp = LastPointOverfitter(gain_scale=10.0, growth="exp")
    exp.fit_one(X0, 2.0)
    assert exp.predict(X0) == pytest.approx(np.exp(0.1) * 2.0)
    with pytest.raises(ValueError):
        LastPointOverfitter(gain_scale=0.0)
    with pytest.raises(ValueError):
        LastPointOverfitter(growth="quadratic")


# ---------------------------------------------------------------------------
# guarantee helpers (Theorems 1/3/4, Remarks)
# ---------------------------------------------------------------------------


def test_bound_helpers_match_theorems() -> None:
    # Theorem 1 / Remark 1: 1 - (2*alpha - 1/(n+1))_+
    assert marginal_coverage_floor(0.1, 9) == pytest.approx(0.9)
    assert marginal_coverage_floor(0.1, 0) == 1.0
    assert marginal_coverage_floor(0.1, 10**6) == pytest.approx(0.8, abs=1e-6)
    # Remark-1 floor decreases to the Theorem-1 limit 1 - 2*alpha as n grows
    assert marginal_coverage_floor(0.1, 9) > marginal_coverage_floor(0.1, 100) > 0.8
    # Theorem 4, Eq. (10a): 1 - alpha*(n+1)/(n-m+2) - 2*sqrt(nu)
    assert stability_coverage_floor(0.1, 1000, 10, 0.0) == pytest.approx(1.0 - 0.1 * 1001.0 / 992.0)
    assert stability_coverage_floor(0.1, 1000, 10, 0.01) == pytest.approx(
        1.0 - 0.1 * 1001.0 / 992.0 - 0.2
    )
    # Theorem 3 / Eq. (7): 2*alpha + sqrt(2*log(...)/(alpha^2 (n+1)))
    b_fix = training_conditional_miscoverage_bound(999, 0.1, 0.05)
    assert b_fix == pytest.approx(0.2 + np.sqrt(2.0 * np.log(20.0) / (0.01 * 1000.0)))
    b_uni = training_conditional_miscoverage_bound(999, 0.1, 0.05, uniform_over_time=True)
    assert b_uni == pytest.approx(0.2 + np.sqrt(2.0 * np.log(1000.0**2 / 0.05) / (0.01 * 1000.0)))
    assert b_uni > b_fix  # uniform-in-time is the price of anytime validity
    assert b_fix > training_conditional_miscoverage_bound(10**6, 0.1, 0.05)  # decreasing
    with pytest.raises(ValueError):
        marginal_coverage_floor(0.0, 5)
    with pytest.raises(ValueError):
        stability_coverage_floor(0.1, 5, 6, 0.0)  # m > n
    with pytest.raises(ValueError):
        stability_coverage_floor(0.1, 5, 1, 1.5)  # nu out of [0, 1]
    with pytest.raises(ValueError):
        training_conditional_miscoverage_bound(5, 0.1, 0.0)
    with pytest.raises(ValueError):
        training_conditional_miscoverage_bound(-1, 0.1, 0.05)


# ---------------------------------------------------------------------------
# Monte-Carlo coverage regimes (seeded SYNTHETIC; tolerances documented)
# ---------------------------------------------------------------------------


def _linear_stream(seed: int, n: int, d: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Paper Sec. 4.1 setup (scaled): X ~ N(0, I_d), theta* = e_1, eps ~ N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] + rng.normal(0.0, sigma, n)
    return X, y


def test_mc_stable_ridge_coverage_near_nominal() -> None:
    """REGIME: STABLE predictor (Theorem 4 / Assumption 1 with small nu).

    Sequential ridge on the paper's Sec. 4.1 i.i.d. linear stream (scaled to
    d=4, n=500, sigma=0.5; the paper uses d=200, n=40000 and reports coverage
    "extremely close to the nominal level"). M=120 streams x H=40 holdout =
    4800 coverage draws, SE ~= sqrt(0.9*0.1/4800) ~= 0.0043; the +/- 0.035
    window around 1 - alpha is ~8 SE and absorbs the mild conservatism from
    early-trajectory score mismatch. Seeds pinned (1000 + k; holdout 501000+).
    """
    covs = []
    for k in range(120):
        X, y = _linear_stream(1000 + k, 500, 4, 0.5)
        Xt, yt = _linear_stream(501_000 + k, 40, 4, 0.5)
        rc = RollingConformal(lambda: OnlineRidge(4, lam=0.5, score="sq"), alpha=ALPHA)
        covs.append(rc.run_stream(X, y, Xt, yt).coverage)
    mean_cov = float(np.mean(covs))
    assert 0.87 <= mean_cov <= 0.94  # approaches 1 - alpha = 0.9, not 1 - 2*alpha
    assert mean_cov >= marginal_coverage_floor(ALPHA, 500) - 0.01  # Theorem 1 floor


def test_mc_stable_trackers_coverage_near_nominal() -> None:
    """REGIME: STABLE predictors (mean and median trackers converge).

    I.i.d. N(0, 1) labels; M=100 x H=30 = 3000 draws per tracker, SE ~= 0.0055;
    window +/- 0.04 around nominal is ~7 SE. Seeds pinned (9000 + k).
    """
    mean_covs, med_covs = [], []
    for k in range(100):
        rng = np.random.default_rng(9000 + k)
        y = rng.normal(size=400)
        rng_h = np.random.default_rng(959_000 + k)
        yh = rng_h.normal(size=30)
        X, Xh = np.zeros((400, 1)), np.zeros((30, 1))
        r1 = RollingConformal(lambda: OnlineMeanTracker(), alpha=ALPHA)
        r1.fit_stream(X, y)
        mean_covs.append(float(np.mean(r1.evaluate(Xh, yh)[1])))
        r2 = RollingConformal(lambda: OnlineQuantileTracker(tau=0.5), alpha=ALPHA)
        r2.fit_stream(X, y)
        med_covs.append(float(np.mean(r2.evaluate(Xh, yh)[1])))
    assert 0.86 <= float(np.mean(mean_covs)) <= 0.95
    assert 0.86 <= float(np.mean(med_covs)) <= 0.95


def test_mc_rolling_narrower_than_frozen_split() -> None:
    """REGIME: STABLE predictor; contrast vs split-CP (paper Sec. 4.1, Fig. 2).

    Same i.i.d. stream for both methods (paper: "using the same data stream
    and independent test point for all methods within each trial"). Split-CP
    freezes the model at m=100 of n=2000 points (their fixed-m setting 1,
    scaled from d=200/m=1000/n=5000/sigma=0.2 to d=40/m=100/n=2000/sigma=0.2:
    the frozen model stays clearly inferior, d/m = 0.4); rolling-CP scores
    every point under its current state and rolls it into training, using ALL
    the data. Paper finding reproduced: rolling sets are narrower at similar
    coverage. M=40 x H=25 = 1000 draws, SE ~= 0.0095 on each coverage; width
    ratio measured 0.86, asserted <= 0.95. Rolling coverage runs slightly
    ABOVE nominal (early-trajectory conservatism) — honest: the guarantee is
    only >= 1 - 2*alpha; nominal is the stable-limit, not a bound.
    Seeds pinned (7000 + k).
    """
    rcov, scov, rwidth, swidth = [], [], [], []
    for k in range(40):
        X, y = _linear_stream(7000 + k, 2000, 40, 0.2)
        Xt, yt = _linear_stream(907_000 + k, 25, 40, 0.2)

        def factory() -> OnlineRidge:
            return OnlineRidge(40, lam=0.1, score="abs")

        r = RollingConformal(factory, alpha=ALPHA).run_stream(X, y, Xt, yt)
        s = SplitConformal(factory, alpha=ALPHA, n_train=100).run_stream(X, y, Xt, yt)
        rcov.append(r.coverage)
        scov.append(s.coverage)
        rwidth.append(r.mean_width)
        swidth.append(s.mean_width)
    ratio = float(np.mean(rwidth) / np.mean(swidth))
    assert ratio <= 0.95  # rolling exploits the models trained on all n points
    assert 0.85 <= float(np.mean(scov)) <= 0.94  # split: exact 1 - alpha validity
    assert 0.87 <= float(np.mean(rcov)) <= 0.97  # rolling: >= floor, near/above nominal


def test_mc_adversarial_fold_attains_factor_two() -> None:
    """REGIME: ADVERSARIAL — worst case of Proposition 1, factor two ATTAINED.

    The paper's App. A.4 construction: i.i.d. Unif(0,1) labels, alternating
    identity/fold scores with lo = 1 - 2*alpha + nu = 0.85, hi = 1, so
    C_n -> [0, 0.85) a.s. and limiting coverage = 0.85 = 1 - 2*alpha + nu.
    M=150 x H=50 = 7500 draws, SE ~= 0.004; measured 0.842 (finite-n DKW
    wobble ~ -0.008 below the limit). Asserts: (i) coverage >= floor
    1 - 2*alpha - 2*SE (the guarantee HOLDS — this is what saves you);
    (ii) coverage <= 0.885, strictly below nominal 0.9 (a 1 - alpha claim
    without stability assumptions is FALSE here). Honesty: 1 - 2*alpha is a
    worst case, not typical — see the STABLE tests above. Seeds pinned (3000+k).
    """
    lo, hi = 0.85, 1.0  # nu = 0.05
    covs = []
    for k in range(150):
        rng = np.random.default_rng(3000 + k)
        y = rng.uniform(0.0, 1.0, 800)
        rng_h = np.random.default_rng(503_000 + k)
        yh = rng_h.uniform(0.0, 1.0, 50)
        rc = RollingConformal(lambda: AlternatingFoldScorer(lo, hi), alpha=ALPHA)
        rc.fit_stream(np.zeros((800, 1)), y)
        covs.append(float(np.mean(rc.evaluate(np.zeros((50, 1)), yh)[1])))
    mean_cov = float(np.mean(covs))
    assert 0.78 <= mean_cov <= 0.885  # floor 0.8 - 2*SE; limit 0.85 < nominal 0.9
    assert mean_cov >= marginal_coverage_floor(ALPHA, 800) - 0.02


@pytest.mark.parametrize(
    ("alpha", "nu"),
    [(0.05, 0.025), (0.2, 0.1)],
)
def test_mc_adversarial_fold_factor_two_scales_with_alpha(alpha: float, nu: float) -> None:
    """REGIME: ADVERSARIAL (Proposition 1) at two more levels.

    Limiting coverage 1 - 2*alpha + nu at n=1500 (finite-n DKW wobble shrinks
    with n); M=120 x H=40 = 4800 draws, SE ~= sqrt(p(1-p)/4800) <= 0.007.
    Window: [floor - 0.02, limit + 0.025]; measured 0.927 (alpha=0.05, limit
    0.925) and 0.702 (alpha=0.2, limit 0.700) — both strictly below nominal
    (<= 1 - alpha - 0.015): the factor-two floor is what remains valid.
    Seeds pinned (4000 + k; holdout 504000 + k).
    """
    lo = 1.0 - 2.0 * alpha + nu
    covs = []
    for k in range(120):
        rng = np.random.default_rng(4000 + k)
        y = rng.uniform(0.0, 1.0, 1500)
        rng_h = np.random.default_rng(504_000 + k)
        yh = rng_h.uniform(0.0, 1.0, 40)
        rc = RollingConformal(lambda: AlternatingFoldScorer(lo, 1.0), alpha=alpha)
        rc.fit_stream(np.zeros((1500, 1)), y)
        covs.append(float(np.mean(rc.evaluate(np.zeros((40, 1)), yh)[1])))
    mean_cov = float(np.mean(covs))
    assert (1.0 - 2.0 * alpha) - 0.02 <= mean_cov <= (1.0 - 2.0 * alpha + nu) + 0.025
    assert mean_cov <= (1.0 - alpha) - 0.015  # below nominal: factor two is needed


@pytest.mark.parametrize("growth", ["linear", "exp"])
def test_mc_unstable_overfitter_naive_split_breaks_rolling_holds(growth: str) -> None:
    """REGIME: UNSTABLE predictor, NOT the worst case — honesty in both directions.

    LastPointOverfitter memorizes only the latest label with a growing gain
    (g_i = 1 + (i-1)/40 linear, or exp((i-1)/40)), so arrival-time scores
    inflate over the stream. The naive 'calibrate at arrival, test with the
    final model' split violates split-CP's fixed-score premise (Eq. (3)) and
    collapses: measured 0.71 (linear) / 0.12 (exp) vs the 1 - 2*alpha = 0.8
    floor it would need — NO guarantee exists for it. Rolling-CP's same-state
    comparisons keep Theorem 1 intact: measured 1.00 — it OVERcovers here.
    Honest reading: instability alone does not attain 1 - 2*alpha (the
    adversarial fold does); the floor is universal, typical behavior is not.
    M=100 x H=30 = 3000 draws, SE ~= 0.007. Seeds pinned (5000 + k).
    """
    roll, naive = [], []
    for k in range(100):
        rng = np.random.default_rng(5000 + k)
        y = rng.normal(size=600)
        rng_h = np.random.default_rng(805_000 + k)
        yh = rng_h.normal(size=30)
        X, Xh = np.zeros((600, 1)), np.zeros((30, 1))
        rc = RollingConformal(lambda: LastPointOverfitter(40.0, growth), alpha=ALPHA)
        rc.fit_stream(X, y)
        roll.append(float(np.mean(rc.evaluate(Xh, yh)[1])))
        naive.append(float(np.mean(rc.naive_evaluate(Xh, yh)[1])))
    mean_roll, mean_naive = float(np.mean(roll)), float(np.mean(naive))
    assert mean_roll >= (1.0 - 2.0 * ALPHA) - 0.02  # Theorem 1 floor holds
    assert mean_naive <= 0.78  # naive arrival-time split breaks below the floor
    assert mean_naive < mean_roll - 0.25  # decisive separation


def test_determinism_pinned() -> None:
    """Same seeds -> bit-identical outputs; independent seeds -> different draws."""

    def run(seed0: int) -> tuple[np.ndarray, np.ndarray, float]:
        out = []
        for k in range(8):
            X, y = _linear_stream(seed0 + k, 120, 3, 0.5)
            Xt, yt = _linear_stream(seed0 + 100_000 + k, 10, 3, 0.5)
            rc = RollingConformal(lambda: OnlineRidge(3, lam=0.2), alpha=ALPHA)
            res = rc.run_stream(X, y, Xt, yt)
            out.append((res.pvalues, res.covered, res.coverage))
        return out[0][0], out[0][1], float(np.mean([o[2] for o in out]))

    p1, c1, m1 = run(42_000)
    p2, c2, m2 = run(42_000)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(c1, c2)
    assert m1 == m2
    p3, _, _ = run(43_000)
    assert not np.array_equal(p1, p3)
