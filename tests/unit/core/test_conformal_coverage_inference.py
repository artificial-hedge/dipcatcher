"""Tests for conformal_coverage_inference: realized-coverage inference under
temporal dependence.

Ported theory under test (Zhai, Cheng & Wu 2026, arXiv:2609.33868; FDM from
Wu 2005a PNAS; block sampling from Zhang, Ho, Wendler & Wu 2013 SPA):
- split-conformal cutoff k_n = ceil((n+1)(1-alpha)) and realized coverage
  Cov_{n,m} (their Eqs. (1)-(2));
- coupled Monte-Carlo estimators of Wu's FDM delta_r(k) / indicator FDM
  theta_k (their Eqs. (3)-(4), Lemma 8) checked against the AR(1) analytic
  value delta_2(k) = sqrt(2) sigma phi^k;
- non-asymptotic marginal coverage bounds (their Theorem 1, Eqs. (5)-(6))
  including the certified-precondition fail-closed path, plus a replication
  of their Study 1 on the NONMIXING Bernoulli recursion;
- block variance estimator for sigma_cov^2 (their Eq. (10), Prop. 4) with
  moving-block and Politis-Romano stationary-bootstrap variants, checked
  against the binomial limit alpha(1-alpha) on i.i.d. scores and against the
  empirical coverage-error SD on AR(1);
- coverage z-test (their Sec. 3, Thm 3) incl. the fixed-B scaled-t reference
  (their Eq. (24), Jones et al. 2006): type-I size near eta, power under a
  deliberately miscalibrated predictor;
- long-memory block sampling (their App. C.1, Thms 6-7): two-scale memory
  exponent beta_hat (their Eq. (37)), normalization R_N(u) (their Eq. (38))
  against its closed form, and block-sampling vs Gaussian-z size in the
  Rosenblatt regime beta = 0.6 where the Gaussian limit fails.

All streams are seeded SYNTHETIC draws (np.random.default_rng, pinned seeds —
determinism, no market data, never market evidence). Documented MC
tolerances: type-I/size windows carry ~3-4 binomial MC standard deviations
plus the finite-sample slack their Table 3 documents (the normal reference
over-rejects at moderate n; the scaled-t reference corrects it); SE-vs-
theory windows carry the batch-means degrees-of-freedom factor (B-1)/B and
the estimated-cutoff effect, both of which vanish only slowly in n. The
long-memory generator is the exact Gaussian linear process of their Study 3,
Y_t = sum_j (1+j)^{-beta} eps_{t-j} (truncated MA(inf), FFT convolution,
burn-in discarded) — quant_fund.models has no fARIMA simulator to reuse;
models/long_memory.gph_estimate is used as an independent persistence
cross-check.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import fftconvolve, lfilter

from quant_fund.metrics.conformal_coverage_inference import (
    BlockSamplingInference,
    CoverageZTest,
    block_sampling_inference,
    coverage_normalization,
    coverage_ztest,
    default_block_length,
    fdm_coupling_norms,
    fdm_transfer_bound,
    indicator_fdm,
    marginal_coverage_bound,
    memory_exponent,
    realized_coverage,
    sigma_cov_squared,
    split_conformal_cutoff,
)
from quant_fund.models.long_memory import gph_estimate

Array = NDArray[np.float64]

SEED = 20260929
ALPHA_COV = 0.1  # nominal coverage 1 - alpha = 0.90, their Study 2/3 default
NOMINAL = 1.0 - ALPHA_COV

# --- documented MC tolerances (see module docstring) -----------------------
FDM_REL_TOL = 0.03  # coupled MC at 20k pairs: ~0.5% observed, 3% is ~6 sigma
IID_VAR_RATIO_LO, IID_VAR_RATIO_HI = 0.90, 1.10  # sigma_hat^2 / alpha(1-alpha)
SE_BINOM_RATIO = (0.85, 1.08)  # mean SE_hat / sqrt(a(1-a)(1/n+1/m))
SE_EMPSD_RATIO = (0.85, 1.10)  # mean SE_hat / empirical coverage-error SD
AR1_TRACK_RATIO = (0.80, 1.15)  # block SE vs realized SD, per persistence
SIZE_T_WINDOW = (0.02, 0.16)  # scaled-t reference size at eta = 0.10
SIZE_NORMAL_MAX = 0.24  # normal reference over-rejects at moderate n (Table 3)
POWER_MIN = 0.80  # miscalibrated predictor, eta = 0.05
STUDY1_COVERAGE = (0.875, 0.925)  # their Table 1: 90.07% at n = 2^11
BETA_HAT_TOL = 0.20  # Lemma 10 bias is o_P(1/log n), slow at testable n
LM_SIZE_BS_MAX = 0.15  # block sampling, Rosenblatt regime, eta = 0.05
LM_SIZE_Z_MIN = 0.18  # Gaussian z-test over-rejects there (their Table 2)
GPH_D_WINDOW = (0.25, 0.55)  # d = 1 - beta = 0.4 for beta = 0.6, GPH-biased


# --- seeded SYNTHETIC generators (test-local, never market data) ------------


def _ar1(rng: np.random.Generator, size: int, phi: float, *, burn: int = 1000) -> Array:
    """Stationary Gaussian AR(1) via lfilter; burn-in discarded."""
    eps = rng.standard_normal(size + burn)
    return np.asarray(lfilter([1.0], [1.0, -phi], eps)[burn:], dtype=float)


def _linear_long_memory(
    rng: np.random.Generator, size: int, beta: float, *, trunc: int = 8192
) -> Array:
    """Their Study 3 process Y_t = sum_{j>=0} (1+j)^{-beta} eps_{t-j}, truncated."""
    coefs = (1.0 + np.arange(trunc + 1, dtype=float)) ** (-beta)
    eps = rng.standard_normal(size + trunc)
    return np.asarray(fftconvolve(eps, coefs)[trunc : trunc + size], dtype=float)


def _coupled_ar1(
    rng: np.random.Generator, phi: float, sigma: float, k_max: int, n_mc: int
) -> tuple[Array, Array]:
    """Paired AR(1) draws sharing every innovation except eps_0 (Wu coupling).

    X_0 = phi*P + eps_0 and X_0* = phi*P + eps_0' with a shared stationary
    remote past P ~ N(0, sigma^2/(1-phi^2)); then X_k - X_k* = phi^k (eps_0 -
    eps_0'), so delta_2(k) = sqrt(2) sigma phi^k in closed form.
    """
    past = rng.standard_normal(n_mc) * sigma / np.sqrt(1.0 - phi * phi)
    eps0 = rng.standard_normal(n_mc) * sigma
    eps0_star = rng.standard_normal(n_mc) * sigma
    shared = rng.standard_normal((n_mc, k_max)) * sigma
    x = np.empty((n_mc, k_max + 1))
    x_star = np.empty((n_mc, k_max + 1))
    x[:, 0] = phi * past + eps0
    x_star[:, 0] = phi * past + eps0_star
    for k in range(1, k_max + 1):
        x[:, k] = phi * x[:, k - 1] + shared[:, k - 1]
        x_star[:, k] = phi * x_star[:, k - 1] + shared[:, k - 1]
    return x, x_star


# ---------------------------------------------------------------------------
# (a) conformal cutoff and realized coverage: exact rank + fail-closed edges
# ---------------------------------------------------------------------------


def test_cutoff_is_exact_conformal_rank() -> None:
    """q_hat_n = k_n-th order statistic with k_n = ceil((n+1)(1-alpha)) (Eq. (1))."""
    rng = np.random.default_rng(SEED)
    v = rng.random(99)
    q = split_conformal_cutoff(v, ALPHA_COV)
    assert q == pytest.approx(float(np.sort(v)[89]), abs=0.0)  # k_99 = ceil(90) = 90
    assert split_conformal_cutoff(v[:9], 0.5) == pytest.approx(float(np.sort(v[:9])[4]))


def test_cutoff_infinite_when_rank_exceeds_n() -> None:
    """Their Eq. (1) convention: k_n > n gives q_hat = inf and coverage 1."""
    v = np.array([0.2, 0.4, 0.6, 0.8])
    assert split_conformal_cutoff(v, ALPHA_COV) == np.inf  # k_4 = ceil(4.5) = 5 > 4
    assert realized_coverage(np.array([1e6, -1e6, 0.0]), np.inf) == 1.0


def test_realized_coverage_matches_manual_fraction() -> None:
    test = np.array([0.1, 0.5, 0.9, 0.5, 0.4])
    assert realized_coverage(test, 0.5) == pytest.approx(0.8)  # 1{V <= 0.5}: 4 of 5
    assert realized_coverage(test, 0.05) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("scores", "alpha"),
    [
        (np.array([]), ALPHA_COV),  # empty
        (np.array([1.0, np.nan]), ALPHA_COV),  # NaN
        (np.array([1.0, 2.0]), 0.0),  # alpha out of (0, 1)
        (np.array([1.0, 2.0]), 1.0),
        (np.array([1.0, 2.0]), np.nan),
    ],
)
def test_cutoff_fail_closed(scores: Array, alpha: float) -> None:
    with pytest.raises(ValueError):
        split_conformal_cutoff(scores, alpha)


@pytest.mark.parametrize("cutoff", [np.nan, -np.inf])
def test_realized_coverage_fail_closed(cutoff: float) -> None:
    with pytest.raises(ValueError):
        realized_coverage(np.array([0.1, 0.2]), cutoff)
    with pytest.raises(ValueError):
        realized_coverage(np.array([]), 0.5)


# ---------------------------------------------------------------------------
# (b) FDM estimators: coupled MC vs the AR(1) analytic value (Wu 2005a)
# ---------------------------------------------------------------------------


def test_fdm_ar1_matches_analytic_prediction_error() -> None:
    """delta_hat_2(k) -> sqrt(2) sigma phi^k: FDM is the k-step coupling error."""
    rng = np.random.default_rng(SEED + 1)
    phi, sigma, k_max, n_mc = 0.8, 1.3, 8, 20_000
    x, x_star = _coupled_ar1(rng, phi, sigma, k_max, n_mc)
    delta_hat = fdm_coupling_norms(x, x_star, r=2.0)
    delta_exact = np.sqrt(2.0) * sigma * phi ** np.arange(k_max + 1)
    assert delta_hat.shape == (k_max + 1,)
    rel_err = np.abs(delta_hat / delta_exact - 1.0)
    assert float(rel_err.max()) < FDM_REL_TOL
    # r = 1 member on the same pairs: E|X_k - X_k*| = phi^k * E|eps0 - eps0'|
    d1 = fdm_coupling_norms(x, x_star, r=1.0)
    assert float(d1[0]) > 0.0 and float(d1[-1]) < float(d1[0])


def test_indicator_fdm_bounded_decaying_and_theta_finite() -> None:
    """theta_hat_k in [0, 1], decaying in k; Theta_hat = sum_k theta_hat_k finite."""
    rng = np.random.default_rng(SEED + 2)
    x, x_star = _coupled_ar1(rng, 0.8, 1.0, 10, 8_000)
    theta = indicator_fdm(np.abs(x), np.abs(x_star))
    assert theta.shape == (11,)
    assert bool(np.all(theta >= 0.0)) and bool(np.all(theta <= 1.0))
    assert float(theta[-1]) < float(theta[0])  # coupling effect decays
    assert bool(np.all(np.diff(theta) < 0.03))  # monotone decay within MC slack
    assert float(theta.sum()) < 11.0  # Theta_hat finite (their Assumption 1 plug-in)


def test_fdm_transfer_bound_lemma8() -> None:
    """Their Eq. (18): theta_k <= min{1, sqrt(1+2M) delta_r^V(k)^(r/(2r+2))}."""
    delta = np.array([0.01, 0.5])
    bound = fdm_transfer_bound(delta, m_bound=2.0, r=2.0)
    expo = 2.0 / 6.0
    expected = np.minimum(1.0, np.sqrt(5.0) * delta**expo)
    assert np.allclose(bound, expected)
    assert float(bound[1]) == 1.0  # clamp at the trivial bound
    with pytest.raises(ValueError):
        fdm_transfer_bound(np.array([-0.1]), m_bound=1.0)
    with pytest.raises(ValueError):
        fdm_transfer_bound(np.array([0.1]), m_bound=-1.0)
    with pytest.raises(ValueError):
        fdm_transfer_bound(np.array([0.1]), m_bound=1.0, r=0.5)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.zeros((4, 3)), np.zeros((4, 4))),  # shape mismatch
        (np.zeros(6), np.zeros(6)),  # 1-D not a coupled pair
        (np.zeros((0, 3)), np.zeros((0, 3))),  # empty
        (np.zeros((4, 3)), np.full((4, 3), np.nan)),  # NaN
    ],
)
def test_fdm_pair_fail_closed(a: Array, b: Array) -> None:
    """Shared coupled-pair validation: both estimators reject malformed input."""
    with pytest.raises(ValueError):
        fdm_coupling_norms(a, b)
    with pytest.raises(ValueError):
        indicator_fdm(a, b)


def test_fdm_coupling_norms_rejects_bad_r() -> None:
    pair = (np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        fdm_coupling_norms(*pair, r=0.5)  # r < 1
    with pytest.raises(ValueError):
        fdm_coupling_norms(*pair, r=np.nan)


# ---------------------------------------------------------------------------
# (c) non-asymptotic marginal coverage bounds (their Theorem 1)
# ---------------------------------------------------------------------------


def test_marginal_coverage_bound_formulas() -> None:
    """Eqs. (5)-(6) evaluated exactly; bounds shrink with n."""
    n, theta, alpha = 1000, 0.3, ALPHA_COV
    out = marginal_coverage_bound(n, theta, alpha_coverage=alpha)
    expected_df = 2.0 / n + 3.0 * (theta**2 / (4.0 * n)) ** (1.0 / 3.0)
    assert out.density_free == pytest.approx(expected_df, rel=1e-12)
    assert out.local_density is None  # density constants not supplied
    out2 = marginal_coverage_bound(
        2000, theta, alpha_coverage=alpha, c_f=0.5, l_lipschitz=2.0, e=0.5
    )
    expected_ld = (
        2.0 * 2.0 / (0.5 * 2000)
        + 2.0 * np.sqrt(2.0) * 2.0 * theta / (0.5 * np.sqrt(2000))
        + 8.0 * theta**2 / (2000 * 0.25 * 0.25)
    )
    assert out2.local_density == pytest.approx(expected_ld, rel=1e-12)
    smaller = marginal_coverage_bound(4000, theta, alpha_coverage=alpha)
    assert smaller.density_free is not None and out.density_free is not None
    assert float(smaller.density_free) < float(out.density_free)  # bound tightens in n


def test_marginal_coverage_bound_precondition_fail_closed() -> None:
    """Uncertified precondition reports None, never a fabricated number."""
    out = marginal_coverage_bound(500, 4.1, alpha_coverage=ALPHA_COV)
    assert out.density_free is None and out.local_density is None
    # 2/n <= c_f*e/2 violated for the local-density branch:
    out2 = marginal_coverage_bound(
        4, 0.01, alpha_coverage=ALPHA_COV, c_f=0.01, l_lipschitz=1.0, e=0.01
    )
    assert out2.local_density is None
    with pytest.raises(ValueError):  # partial density-constant triple
        marginal_coverage_bound(1000, 0.3, alpha_coverage=ALPHA_COV, c_f=0.5)
    with pytest.raises(ValueError):
        marginal_coverage_bound(0, 0.3, alpha_coverage=ALPHA_COV)
    with pytest.raises(ValueError):
        marginal_coverage_bound(1000, -0.1, alpha_coverage=ALPHA_COV)
    with pytest.raises(ValueError):
        marginal_coverage_bound(
            1000, 0.3, alpha_coverage=ALPHA_COV, c_f=-1.0, l_lipschitz=1.0, e=1.0
        )


def test_marginal_coverage_nonmixing_study1() -> None:
    """Their Study 1: nominal 90% is attained on the NONMIXING Bernoulli recursion.

    U_t = (U_{t-1} + B_t)/2, B_t ~ iid Bernoulli(1/2), U_0 ~ Unif(0,1),
    Y_t = U_t^2, V_t = |Y_t - 1/3|: scores fail strong mixing (a distant
    future score determines the present one) yet the indicator FDM decays
    geometrically, so Theorem 1 applies. Their Table 1 reports 90.07% at
    n = 2^11; seeded tolerance is ~3 MC sd around 90% (R = 1500 replicates).
    """
    rng = np.random.default_rng(SEED + 16)
    n, reps = 2048, 1500
    u = rng.random(reps)
    vs = np.empty((reps, n + 1))
    for t in range(n + 1):
        u = (u + (rng.random(reps) < 0.5).astype(float)) / 2.0
        vs[:, t] = np.abs(u * u - 1.0 / 3.0)
    cutoffs = np.array([split_conformal_cutoff(vs[r, :n], ALPHA_COV) for r in range(reps)])
    cov = float(np.mean(vs[:, n] <= cutoffs))
    assert STUDY1_COVERAGE[0] <= cov <= STUDY1_COVERAGE[1]


# ---------------------------------------------------------------------------
# (d) block variance estimator: hand-computed values + fail-closed edges
# ---------------------------------------------------------------------------


def test_sigma_cov_squared_hand_computed() -> None:
    """Eq. (10) exactly on a fixed indicator pattern; moving member likewise."""
    ind = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    nv = sigma_cov_squared(ind, method="nonoverlapping", block=2)
    # block means [1, 0, 1, 0]: (l/B) sum (Ibar_j - Ibar)^2 = 2 * 0.25
    assert nv.sigma2 == pytest.approx(0.5)
    assert (nv.block_length, nv.n_blocks, nv.method) == (2, 4, "nonoverlapping")
    mv = sigma_cov_squared(ind, method="moving", block=2)
    # overlapping batches [1,.5,0,.5,1,.5,0]: l * var0 = 2 * (1/7)
    assert mv.sigma2 == pytest.approx(2.0 / 7.0)
    assert (mv.block_length, mv.n_blocks) == (2, 7)
    nv4 = sigma_cov_squared(ind, method="nonoverlapping", block=4)
    assert nv4.sigma2 == pytest.approx(0.0)  # block means [0.5, 0.5]: degenerate
    assert nv4.sigma == pytest.approx(0.0)
    assert default_block_length(512) == int(np.floor(512 ** (2.0 / 3.0)))


def test_sigma_cov_squared_stationary_member_runs() -> None:
    """Politis-Romano member returns a finite positive LR-variance estimate."""
    rng = np.random.default_rng(SEED + 3)
    ind = (np.abs(_ar1(rng, 600, 0.6)) <= 1.0).astype(float)
    bv = sigma_cov_squared(ind, method="stationary", n_boot=199, seed=11)
    assert np.isfinite(bv.sigma2) and bv.sigma2 > 0.0
    assert bv.method == "stationary" and bv.n_blocks == 199


@pytest.mark.parametrize(
    ("ind", "kwargs"),
    [
        (np.array([1.0, 0.0, 1.0]), {}),  # n < 4
        (np.ones(10), {"block": 10}),  # block >= n
        (np.ones(10), {"block": 0}),  # block < 1
        (np.full(10, 0.5), {}),  # non-binary indicators
        (np.array([1.0, np.nan, 0.0, 1.0]), {}),  # NaN
        (np.ones(10), {"method": "bogus"}),  # unknown method
        (np.ones(10), {"method": "stationary", "n_boot": 1}),  # n_boot < 2
    ],
)
def test_sigma_cov_squared_fail_closed(ind: Array, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sigma_cov_squared(ind, **kwargs)


# ---------------------------------------------------------------------------
# (e) long-memory normalization + memory exponent (their Eqs. (37)-(38))
# ---------------------------------------------------------------------------


def test_coverage_normalization_closed_form_and_continuity() -> None:
    """Eq. (38) via expm1: exact at the critical exponent, continuous across it."""
    n = 1000
    critical = coverage_normalization(0.75, n)
    assert critical == pytest.approx(np.sqrt((1.0 + np.log(n)) / n), rel=1e-12)
    for u in (0.6, 0.9, 1.0):
        raw = np.sqrt((1.0 + (n ** (3.0 - 4.0 * u) - 1.0) / (3.0 - 4.0 * u)) / n)
        assert coverage_normalization(u, n) == pytest.approx(float(raw), rel=1e-10)
    assert abs(coverage_normalization(0.75 + 1e-9, n) - critical) < 1e-6 * critical
    # stronger memory -> larger normalization; stronger persistence of scale
    assert coverage_normalization(0.6, n) > critical > coverage_normalization(0.9, n)
    # short-memory endpoint degrades to the root-n scale: R_N(1) ~ sqrt(2/N)
    assert coverage_normalization(1.0, n) * np.sqrt(n) == pytest.approx(np.sqrt(2.0), rel=0.02)
    with pytest.raises(ValueError):
        coverage_normalization(0.7, 1)
    with pytest.raises(ValueError):
        coverage_normalization(np.nan, 100)


def test_memory_exponent_fallback_and_validation() -> None:
    """Zero block variance -> their stated beta_hat = 3/4 convention; h checks."""
    beta_hat, h_used = memory_exponent(np.zeros(200))
    assert beta_hat == 0.75 and h_used == int(np.floor(np.sqrt(200)))
    with pytest.raises(ValueError):
        memory_exponent(np.zeros(10), h=6)  # 2h > n
    with pytest.raises(ValueError):
        memory_exponent(np.zeros(100), h=0)
    with pytest.raises(ValueError):
        memory_exponent(np.array([1.0, np.nan]))


def test_memory_exponent_recovers_beta_regimes() -> None:
    """beta_hat on their Study 3 process: ordered across regimes, |bias| bounded.

    Lemma 10 gives beta_hat - beta = o_P(1/log n) — slow, so the tolerance is
    the documented BETA_HAT_TOL window at n = 16384 with 6 seeds per beta.
    """
    means: dict[float, float] = {}
    for beta in (0.6, 0.75, 0.9):
        acc = []
        for s in range(6):
            rng = np.random.default_rng(SEED + 20 + s)
            y = _linear_long_memory(rng, 16384, beta)
            acc.append(memory_exponent(y)[0])
        means[beta] = float(np.mean(acc))
        assert abs(means[beta] - beta) <= BETA_HAT_TOL
    assert means[0.6] < means[0.75] < means[0.9]
    assert means[0.6] < 0.75  # Rosenblatt regime correctly identified in the mean


def test_generated_long_memory_stream_is_long_memory() -> None:
    """Independent cross-check: GPH (models/long_memory) sees d = 1 - beta > 0."""
    ds = []
    for s in range(4):
        rng = np.random.default_rng(SEED + 30 + s)
        y = _linear_long_memory(rng, 16384, 0.6)
        ds.append(float(gph_estimate(y)["d"]))
    mean_d = float(np.mean(ds))
    assert GPH_D_WINDOW[0] <= mean_d <= GPH_D_WINDOW[1]
    assert all(d > 0.15 for d in ds)  # persistent, unlike i.i.d. (d = 0)


# ---------------------------------------------------------------------------
# (f) i.i.d. sanity: block SE -> binomial sqrt(p(1-p)(1/n+1/m))
# ---------------------------------------------------------------------------


def test_iid_sigma_cov_matches_binomial_variance() -> None:
    """Under i.i.d. scores sigma_cov^2 = alpha(1-alpha); the estimator is
    asymptotically unbiased there (batch-means dof (B-1)/B vanishes slowly,
    hence the large n and the documented window)."""
    rng = np.random.default_rng(SEED + 10)
    n, reps = 20_000, 60
    est = []
    for _ in range(reps):
        v = np.abs(rng.standard_normal(n))
        q = split_conformal_cutoff(v, ALPHA_COV)
        est.append(sigma_cov_squared((v <= q).astype(float)).sigma2)
    ratio = float(np.mean(est)) / (ALPHA_COV * NOMINAL)
    assert IID_VAR_RATIO_LO <= ratio <= IID_VAR_RATIO_HI
    # stationary-bootstrap member on the same footing (Politis-White block)
    rng2 = np.random.default_rng(SEED + 10)
    est2 = []
    for _ in range(40):
        v = np.abs(rng2.standard_normal(4000))
        q = split_conformal_cutoff(v, ALPHA_COV)
        est2.append(
            sigma_cov_squared(
                (v <= q).astype(float), method="stationary", n_boot=299, seed=5
            ).sigma2
        )
    ratio2 = float(np.mean(est2)) / (ALPHA_COV * NOMINAL)
    assert 0.85 <= ratio2 <= 1.15


def test_iid_se_matches_binomial_and_empirical_sd() -> None:
    """SE_hat tracks BOTH the binomial formula and the realized coverage SD."""
    rng = np.random.default_rng(SEED + 11)
    n = m = 4000
    reps = 200
    ses, covs = [], []
    for _ in range(reps):
        v = np.abs(rng.standard_normal(n + m))
        r = coverage_ztest(v[:n], v[n:], alpha_coverage=ALPHA_COV)
        ses.append(r.se)
        covs.append(r.realized_coverage)
    binom_se = float(np.sqrt(ALPHA_COV * NOMINAL * (1.0 / n + 1.0 / m)))
    emp_sd = float(np.std(covs, ddof=1))
    assert SE_BINOM_RATIO[0] <= float(np.mean(ses)) / binom_se <= SE_BINOM_RATIO[1]
    assert SE_EMPSD_RATIO[0] <= float(np.mean(ses)) / emp_sd <= SE_EMPSD_RATIO[1]


# ---------------------------------------------------------------------------
# (g) AR(1): coverage-error SD grows with persistence and the block SE tracks it
# ---------------------------------------------------------------------------


def test_ar1_coverage_error_sd_grows_with_persistence() -> None:
    """Their Sec. 3: accumulated lagged indicator covariance inflates sigma_cov.

    Var[sqrt(nm/(n+m)) (Cov - (1-alpha))] and the block estimate both increase
    with phi; at phi = 0.9 the variance estimate is >2x the i.i.d. level.
    """
    n = m = 1500
    reps = 200
    mean_s2, emp_var, track = {}, {}, {}
    for phi in (0.0, 0.6, 0.9):
        rng = np.random.default_rng(SEED + 12)
        s2, errs = [], []
        for _ in range(reps):
            v = np.abs(_ar1(rng, n + m, phi))
            q = split_conformal_cutoff(v[:n], ALPHA_COV)
            s2.append(sigma_cov_squared((v[:n] <= q).astype(float)).sigma2)
            cov = realized_coverage(v[n:], q)
            errs.append(np.sqrt(n * m / (n + m)) * (cov - NOMINAL))
        mean_s2[phi] = float(np.mean(s2))
        emp_var[phi] = float(np.var(errs, ddof=1))
        track[phi] = float(np.sqrt(np.mean(s2)) / np.std(errs, ddof=1))
    assert mean_s2[0.0] < mean_s2[0.6] < mean_s2[0.9]
    assert emp_var[0.0] < emp_var[0.6] < emp_var[0.9]
    assert mean_s2[0.9] > 2.0 * mean_s2[0.0]
    for phi in (0.0, 0.6, 0.9):
        assert AR1_TRACK_RATIO[0] <= track[phi] <= AR1_TRACK_RATIO[1]
    # i.i.d. anchor: at phi = 0 the LR variance sits at the binomial level
    assert mean_s2[0.0] == pytest.approx(ALPHA_COV * NOMINAL, rel=0.15)


# ---------------------------------------------------------------------------
# (h) coverage z-test: size, power, CI/PI consistency, determinism, edges
# ---------------------------------------------------------------------------


def _size_windows(rng_seed: int, gen, n: int, reps: int, eta: float) -> tuple[float, float]:
    """(normal-reference, scaled-t-reference) two-sided rejection rates."""
    rng = np.random.default_rng(rng_seed)
    rej_normal = rej_t = 0
    for _ in range(reps):
        v = np.abs(gen(rng, 2 * n))
        rn = coverage_ztest(v[:n], v[n:], alpha_coverage=ALPHA_COV, eta=eta)
        rt = coverage_ztest(v[:n], v[n:], alpha_coverage=ALPHA_COV, eta=eta, reference="student_t")
        rej_normal += int(rn.p_value < eta)
        rej_t += int(rt.p_value < eta)
    return rej_normal / reps, rej_t / reps


def test_ztest_size_iid_normal_and_corrected_t() -> None:
    """Type-I size at eta = 0.10 on i.i.d. scores, n = m = 4000.

    Their Table 3 pattern: the normal reference mildly over-rejects at
    moderate n; the fixed-B scaled-t reference (their Eq. (24)) corrects it.
    """
    size_normal, size_t = _size_windows(
        SEED + 13, lambda r, t: r.standard_normal(t), 4000, 200, 0.10
    )
    assert SIZE_T_WINDOW[0] <= size_t <= SIZE_T_WINDOW[1]
    assert 0.05 <= size_normal <= SIZE_NORMAL_MAX
    assert size_t <= size_normal + 0.02


def test_ztest_size_ar1_dependent_scores() -> None:
    """Size under AR(1) phi = 0.8 dependence (their Assumption 1 class)."""
    size_normal, size_t = _size_windows(SEED + 14, lambda r, t: _ar1(r, t, 0.8), 1500, 200, 0.10)
    assert SIZE_T_WINDOW[0] <= size_t <= 0.16
    assert size_normal <= 0.26


def test_ztest_power_under_miscalibration() -> None:
    """Deliberately miscalibrated predictor: calibration window volatility
    x0.7 => true coverage far below nominal => high rejection rate at 5%."""
    rng = np.random.default_rng(SEED + 15)
    n = m = 600
    reps = 100
    rej = 0
    for _ in range(reps):
        y = _ar1(rng, n + m, 0.7)
        cal = np.abs(y[:n]) * 0.7  # intervals calibrated on a quiet window
        r = coverage_ztest(cal, np.abs(y[n:]), alpha_coverage=ALPHA_COV, eta=0.05)
        assert r.statistic < 0.0  # systematic undercoverage
        rej += int(r.p_value < 0.05)
    assert rej / reps >= POWER_MIN


def test_ztest_ci_pi_consistency_and_bounds() -> None:
    """Wald CI contains the nominal iff the two-sided test does not reject;
    the pre-test PI is centered on the nominal; endpoints stay in [0, 1]."""
    rng = np.random.default_rng(SEED + 17)
    checked_reject = checked_accept = 0
    for i in range(24):
        phi = 0.7 if i % 2 == 0 else 0.0
        y = _ar1(rng, 1200, phi)
        scale = 0.6 if i % 3 == 0 else 1.0  # mix in miscalibrated runs
        cal, test = np.abs(y[:600]) * scale, np.abs(y[600:])
        r = coverage_ztest(cal, test, alpha_coverage=ALPHA_COV, eta=0.05)
        assert 0.0 <= r.ci_low <= r.ci_high <= 1.0
        assert 0.0 <= r.pi_low <= r.nominal <= r.pi_high <= 1.0
        assert r.pi_high - r.pi_low == pytest.approx(r.ci_high - r.ci_low)
        contains = r.ci_low <= r.nominal <= r.ci_high
        assert contains == (r.p_value >= 0.05)
        checked_accept += int(contains)
        checked_reject += int(not contains)
    assert checked_accept > 0 and checked_reject > 0  # both branches exercised


def test_ztest_determinism_pinned() -> None:
    """Same inputs and seed => bitwise-identical output (all members)."""
    rng = np.random.default_rng(SEED + 18)
    y = _ar1(rng, 1000, 0.6)
    v = np.abs(y)
    for method in ("nonoverlapping", "moving", "stationary"):
        r1 = coverage_ztest(v[:500], v[500:], method=method, seed=123)
        r2 = coverage_ztest(v[:500], v[500:], method=method, seed=123)
        assert r1 == r2
        assert isinstance(r1, CoverageZTest)
    r3 = coverage_ztest(v[:500], v[500:], method="stationary", seed=124)
    assert r3.sigma_cov != r1.sigma_cov  # a different bootstrap seed moves the estimate


def test_ztest_fail_closed() -> None:
    """Infinite cutoff, zero block variance, invalid reference combos raise."""
    rng = np.random.default_rng(SEED + 19)
    v = np.abs(rng.standard_normal(400))
    with pytest.raises(ValueError):  # k_n > n: infinite cutoff is untestable
        coverage_ztest(np.array([0.1, 0.2, 0.3, 0.4]), v[:10], alpha_coverage=ALPHA_COV)
    with pytest.raises(ValueError):  # constant calibration scores: sigma_hat^2 = 0
        coverage_ztest(np.ones(400), v[:10], alpha_coverage=ALPHA_COV)
    with pytest.raises(ValueError):  # scaled-t reference needs their Eq. (10) blocks
        coverage_ztest(v[:200], v[200:], method="moving", reference="student_t")
    with pytest.raises(ValueError):
        coverage_ztest(v[:200], v[200:], reference="bogus")
    with pytest.raises(ValueError):
        coverage_ztest(v[:200], v[200:], eta=0.0)
    with pytest.raises(ValueError):
        coverage_ztest(v[:200], v[200:], alpha_coverage=1.2)


# ---------------------------------------------------------------------------
# (i) long memory: block sampling beats the Gaussian z-test (their Table 2)
# ---------------------------------------------------------------------------


def test_block_sampling_beats_ztest_in_rosenblatt_regime() -> None:
    """beta = 0.6 (1/2 < beta < 3/4): the Gaussian limit FAILS (their Thm 6).

    Block sampling with estimated normalization keeps the two-sided size at
    eta = 0.05 while the block-SE Gaussian z-test over-rejects — the paper's
    Table 2 pattern (BS near nominal, MBB/HAC far above), here against the
    z-test member at n = m = 4096.
    """
    rng = np.random.default_rng(SEED + 31)
    n = m = 4096
    reps = 40
    rej_bs = rej_z = regimes = 0
    beta_hats = []
    for _ in range(reps):
        y = _linear_long_memory(rng, n + m, 0.6)
        v = np.abs(y)
        bs = block_sampling_inference(v[:n], v[n:], y[:n], alpha_coverage=ALPHA_COV, eta=0.05)
        zt = coverage_ztest(v[:n], v[n:], alpha_coverage=ALPHA_COV, eta=0.05)
        rej_bs += int(bs.reject)
        rej_z += int(zt.p_value < 0.05)
        regimes += int(bs.regime == "rosenblatt")
        beta_hats.append(bs.beta_hat)
    assert rej_bs / reps <= LM_SIZE_BS_MAX
    assert rej_z / reps >= LM_SIZE_Z_MIN
    assert rej_bs < rej_z
    assert float(np.mean(beta_hats)) < 0.75
    assert regimes >= int(0.75 * reps)


def test_block_sampling_fields_and_determinism() -> None:
    """Paired-block geometry, normalization order, p-value/CI coherence."""
    rng = np.random.default_rng(SEED + 32)
    y = _ar1(rng, 512 + 256, 0.7)
    v = np.abs(y)
    bs = block_sampling_inference(v[:512], v[512:], y[:512], alpha_coverage=ALPHA_COV, eta=0.05)
    assert isinstance(bs, BlockSamplingInference)
    b = int(np.floor(512 ** (2.0 / 3.0)))  # 63 in floating point — pinned formula
    assert bs.block_length == b
    assert bs.eval_length == int(np.floor(b * 256 / 512))
    assert bs.n_block_stats == 512 - b - bs.eval_length + 1
    assert bs.r_block > bs.r_full > 0.0  # R_N(u) shrinks with N
    assert bs.statistic == pytest.approx((bs.realized_coverage - bs.nominal) / bs.r_full)
    assert 0.0 <= bs.ci_low <= bs.ci_high <= 1.0
    assert 0.0 <= bs.p_value <= 1.0
    # quantile-based rejection is coherent with the reported interval: the
    # statistic lies outside [c_eta/2, c_1-eta/2] iff realized coverage lies
    # outside [ci_low, ci_high] (clipping cannot break the equivalence: the
    # clipped side is unreachable for a proportion-valued statistic).
    inside = bs.ci_low <= bs.realized_coverage <= bs.ci_high
    assert bs.reject == (not inside)
    again = block_sampling_inference(v[:512], v[512:], y[:512], alpha_coverage=ALPHA_COV, eta=0.05)
    assert bs == again  # fully deterministic, no resampling randomness
    zeta_big = block_sampling_inference(
        v[:512], v[512:], y[:512], alpha_coverage=ALPHA_COV, zeta=0.8
    )
    assert zeta_big.block_length > bs.block_length


def test_block_sampling_fail_closed() -> None:
    """Short streams, misaligned residuals, extreme levels, bad knobs raise."""
    rng = np.random.default_rng(SEED + 33)
    y = _ar1(rng, 600, 0.5)
    v = np.abs(y)
    ok_cal, ok_test, ok_res = v[:300], v[300:400], y[:300]
    with pytest.raises(ValueError):  # fewer than 10 block statistics
        block_sampling_inference(v[:16], v[16:32], y[:16])
    with pytest.raises(ValueError):  # residuals misaligned with calibration
        block_sampling_inference(ok_cal, ok_test, y[:299])
    with pytest.raises(ValueError):  # k_b > b: level too extreme for the blocks
        block_sampling_inference(v[:64], v[64:128], y[:64], alpha_coverage=0.01)
    with pytest.raises(ValueError):  # evaluation length floors to 0
        block_sampling_inference(v[:512], v[512:513], y[:512])
    with pytest.raises(ValueError):
        block_sampling_inference(ok_cal, ok_test, ok_res, zeta=1.5)
    with pytest.raises(ValueError):
        block_sampling_inference(ok_cal, ok_test, ok_res, zeta=0.0)
    with pytest.raises(ValueError):
        block_sampling_inference(ok_cal, ok_test, ok_res, eta=0.0)
    with pytest.raises(ValueError):
        block_sampling_inference(ok_cal, np.array([]), ok_res)
