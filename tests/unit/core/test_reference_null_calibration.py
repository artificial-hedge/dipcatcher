"""Tests for reference_null_calibration: reference-null calibrated e-process thresholds.

Construction under test (Ding, Wei, Zhu & Dai 2026, "Reference-Null Calibrated
Thresholds for E-Processes with Applications to Conformal Martingales",
arXiv:2609.32678):
- the rank/order-statistic calibrated threshold c_hat = k-th smallest of B
  exchangeable reference-null path statistics, k = ceil((1-alpha)(B+1))
  (their Thms 2.1/2.2, Algorithm 1);
- KT histogram betting on conformal p-values (their Sec. 3.2, Alg. 2);
- the restart-mixture e-process (their Sec. 3.3, Eq. 5, Alg. 3);
- the conformal specialization calibrated on i.i.d.-uniform reference banks
  (their pivotal null law, Sec. 3.1/3.4).

Composition: the test p-value streams come from
quant_fund.metrics.conformal_martingale.conformal_p_values (its sequential
smoothed p-values equal the paper's Eq. 2); the Ville crossing check is
cross-validated against conformal_martingale.martingale_alarm and
e_detectors.alarm_threshold — those modules are NOT modified.

Every stream is seeded SYNTHETIC (np.random.default_rng, pinned seeds,
disjoint seed ranges for scores / randomizers / reference banks so the bank
is independent of the test data): correctness checks of the error control,
never market evidence. Monte-Carlo tolerances follow the repo convention
(binomial slack, cf. test_e_detectors / test_confidence_sequences). The
Theorem 2.2 guarantee is UNCONDITIONAL — it includes the calibration-bank
randomness — and a single bank's conditional crossing rate fluctuates at
O(1/sqrt(B)) (the order statistic is Beta(k, B+1-k) on the null CDF scale),
so the type-I tests follow the paper's own multi-bank protocol (their
Sec. 5.2: rejection probability averaged over independently estimated
calibration boundaries on shared fresh null paths): the bank-AVERAGED FA
rate must stay <= alpha + FA_TOL, and every per-bank rate <= alpha +
BANK_TOL with the wider documented bank-conditional slack.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.conformal_martingale import conformal_p_values, martingale_alarm
from quant_fund.metrics.e_detectors import alarm_threshold
from quant_fund.metrics.reference_null_calibration import (
    CalibratedThreshold,
    calibrate_conformal_martingale,
    calibrate_e_process,
    conformal_null_path_maxima,
    crossing_time,
    detection_delay_summary,
    kt_betting_factors,
    kt_histogram_martingale,
    min_reference_nulls,
    normalized_path_maxima,
    reference_null_threshold,
    restart_mixture_kt_martingale,
    restart_weights,
)

Array = NDArray[np.float64]

SEED = 20260929
ALPHA = 0.05
VILLE = 1.0 / ALPHA
N_BINS = 20
# Disjoint seed ranges: scores, p-value randomizers, and reference banks must
# be independent streams (exchangeability of test path and bank, Thm 2.2).
SCORE_OFFSET = 100_000
RANDOMIZER_OFFSET = 200_000
SHIFT_SCORE_OFFSET = 300_000
SHIFT_RANDOMIZER_OFFSET = 400_000
BANK_OFFSET = 900_000
DELAY_BANK_SEED = SEED + 951_000

FA_TOL = 0.02  # binomial slack on the bank-averaged rate (repo convention)
BANK_TOL = 0.035  # per-bank slack: conditional rate ~ Beta(k, B+1-k), sd ~ 1/sqrt(B)

T_FA = 250  # paper Sec. 5.1 shortest main horizon
N_BANKS_KT = 8
N_BANKS_RKT = 4
BANK_KT = 499  # paper Sec. 5.1 bank size K
BANK_RKT = 299
N_EVAL_KT = 500
N_EVAL_RKT = 240

T_DELAY = 400
TAU_DELAY = 200  # paper design: tau = T/2
MU_DELAY = 1.0  # paper signal grid point mu = 1.00
N_REPS_DELAY = 200
BANK_DELAY = 499


def _null_p_streams(n: int, t_len: int, construction_seed_base: int = SEED) -> list[Array]:
    """SYNTHETIC exchangeable null: iid N(0,1) scores -> conformal p-values."""
    out = []
    for rep in range(n):
        rng = np.random.default_rng(construction_seed_base + SCORE_OFFSET + rep)
        scores = rng.standard_normal(t_len)
        out.append(
            conformal_p_values(scores, seed=construction_seed_base + RANDOMIZER_OFFSET + rep)
        )
    return out


def _eval_paths(construction: str, n: int, t_len: int) -> Array:
    streams = _null_p_streams(n, t_len)
    fn = kt_histogram_martingale if construction == "kt" else restart_mixture_kt_martingale
    return np.stack([fn(p, N_BINS) for p in streams])


# ---------------------------------------------------------------------------
# (a) rank calibration core: exact order statistic, minimum B, fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("alpha", "b_min"), [(0.05, 19), (0.01, 99), (0.1, 9)])
def test_min_reference_nulls_documented_minimum(alpha: float, b_min: int) -> None:
    """B_min = ceil(1/alpha - 1); below it Algorithm 1 would set c_hat = +inf."""
    assert min_reference_nulls(alpha) == b_min
    refs = np.arange(1, b_min + 1, dtype=float)
    ok = reference_null_threshold(refs, alpha)
    assert ok.threshold == float(b_min)  # k = B -> the largest order statistic
    with pytest.raises(ValueError, match="too few"):
        reference_null_threshold(refs[:-1], alpha)


def test_reference_null_threshold_exact_order_statistic() -> None:
    """c_hat = k-th smallest reference statistic, k = ceil((1-alpha)(B+1))."""
    refs = np.arange(1, 101, dtype=float)  # B = 100 -> k = ceil(0.95*101) = 96
    out = reference_null_threshold(refs, ALPHA)
    assert isinstance(out, CalibratedThreshold)
    assert out.threshold == 96.0
    assert out.k_order == 96
    assert out.n_reference == 100
    assert out.ville_threshold == VILLE
    assert out.sharpness == pytest.approx(96.0 * ALPHA)


def test_reference_null_threshold_paper_rank_case() -> None:
    """Paper Sec. 6.4: B = 4999, alpha = 0.05 -> the 4750th order statistic."""
    refs = np.arange(1, 5000, dtype=float)
    out = reference_null_threshold(refs, ALPHA)
    assert out.k_order == 4750
    assert out.threshold == 4750.0
    # exactly B - k = 249 reference maxima exceed under strict crossing (Sec. 6.6)
    assert int(np.count_nonzero(refs > out.threshold)) == 249


def test_reference_null_threshold_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        reference_null_threshold(np.array([]), ALPHA)
    with pytest.raises(ValueError, match="nonnegative"):
        reference_null_threshold(np.array([-1.0, 2.0, 3.0] * 10), ALPHA)
    with pytest.raises(ValueError, match="finite"):
        reference_null_threshold(np.full(50, np.nan), ALPHA)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            reference_null_threshold(np.arange(1.0, 200.0), bad_alpha)


def test_normalized_path_maxima_and_template() -> None:
    """R_g = max_t M_t / g_t; the default template is the constant g = 1."""
    paths = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(normalized_path_maxima(paths), [3.0, 6.0])
    g = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(normalized_path_maxima(paths, g), [1.0, 4.0])
    with pytest.raises(ValueError, match="positive"):
        normalized_path_maxima(paths, np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="same length"):
        normalized_path_maxima(paths, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="2-D"):
        normalized_path_maxima(np.array([1.0, 2.0]))


def test_calibrate_e_process_generic_algorithm1() -> None:
    """Algorithm 1 on a generic construction: Gaussian one-step e-values."""
    rng = np.random.default_rng(SEED + 5)
    data = rng.standard_normal((120, 80))  # SYNTHETIC standard-normal nulls

    def gauss_e_process(x: Array) -> Array:
        steps = np.exp(x - 0.5)  # E[e^{X-1/2}] = 1 under N(0,1)
        return np.concatenate([[1.0], np.cumprod(steps)])

    out = calibrate_e_process(data, gauss_e_process, ALPHA)
    assert out.threshold >= 1.0  # every path max >= M_0 = 1
    manual_paths = np.stack([gauss_e_process(data[b]) for b in range(data.shape[0])])
    manual = reference_null_threshold(normalized_path_maxima(manual_paths), ALPHA)
    assert out.threshold == manual.threshold
    # deterministic given the inputs
    again = calibrate_e_process(data, gauss_e_process, ALPHA)
    assert again.threshold == out.threshold

    with pytest.raises(ValueError, match="length >= 2"):
        calibrate_e_process(data, lambda x: np.array([1.0]), ALPHA)
    with pytest.raises(ValueError, match="finite"):
        calibrate_e_process(data, lambda x: np.full(x.size + 1, np.nan), ALPHA)


# ---------------------------------------------------------------------------
# (b) KT histogram betting: smoothing sanity, martingale property
# ---------------------------------------------------------------------------


def test_kt_first_factor_one_and_first_visit_le_one() -> None:
    """KT starts exactly uniform; a first visit to an empty bin cannot explode.

    The half-count smoothing gives factor J/(2(t-1)+J) <= 1 for any bin with
    zero prior count (equality only at t=1) — zero-count bins are damped,
    never explosive (paper Sec. 3.2).
    """
    assert kt_betting_factors(np.array([0.0]), 20)[0] == 1.0
    assert kt_betting_factors(np.array([1.0]), 7)[0] == 1.0
    assert kt_betting_factors(np.array([0.37]), 20)[0] == 1.0
    rng = np.random.default_rng(SEED + 6)
    p = rng.random(300)
    factors = kt_betting_factors(p, N_BINS)
    seen = set()
    for t, pi in enumerate(p):
        j = min(int(pi * N_BINS), N_BINS - 1)
        if j not in seen:
            assert factors[t] <= 1.0 + 1e-12  # first visit to a zero-count bin
        seen.add(j)
    assert bool(np.all(factors > 0.0)) and bool(np.all(np.isfinite(factors)))


def test_kt_zero_count_bins_do_not_explode() -> None:
    """15 of 20 bins stay empty forever: the wealth path stays finite."""
    p = np.array([0.05, 0.25, 0.45, 0.65, 0.85] * 40)  # 5 occupied bins, J=20
    path = kt_histogram_martingale(p, N_BINS)
    assert bool(np.all(np.isfinite(path))) and bool(np.all(path > 0.0))
    mix = restart_mixture_kt_martingale(p, N_BINS)
    assert bool(np.all(np.isfinite(mix))) and bool(np.all(mix > 0.0))


def test_kt_wealth_overflow_is_capped() -> None:
    """Persistent one-bin drift saturates at _E_MAX = 1e300 instead of inf."""
    p = np.full(300, 0.02)  # every p-value in bin 0: wealth ~ J^t
    path = kt_histogram_martingale(p, N_BINS)
    assert bool(np.all(np.isfinite(path)))
    assert float(path.max()) <= 1e300
    mix = restart_mixture_kt_martingale(p, N_BINS)
    assert bool(np.all(np.isfinite(mix)))
    assert float(mix.max()) <= 1e300


def test_kt_path_shape_positivity_determinism() -> None:
    rng = np.random.default_rng(SEED + 7)
    p = rng.random(120)
    path = kt_histogram_martingale(p, N_BINS)
    assert path.shape == (121,)
    assert path[0] == 1.0
    assert bool(np.all(path > 0.0)) and bool(np.all(np.isfinite(path)))
    np.testing.assert_array_equal(path, kt_histogram_martingale(p, N_BINS))
    np.testing.assert_allclose(path[1:], np.cumprod(kt_betting_factors(p, N_BINS)))


def test_kt_fail_closed_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        kt_histogram_martingale(np.array([]), N_BINS)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        kt_histogram_martingale(np.array([0.5, 1.2]), N_BINS)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        kt_histogram_martingale(np.array([0.5, np.nan]), N_BINS)
    with pytest.raises(ValueError, match="n_bins"):
        kt_histogram_martingale(np.array([0.5, 0.5]), 1)


def test_kt_martingale_mean_one_under_null() -> None:
    """E[M_T] = 1 under i.i.d. uniforms (Prop. 3.2), small-T MC check.

    The null wealth is right-skewed (typical paths decay like the KT regret
    while rare paths carry the mean), so the check uses short horizons where
    the Monte-Carlo mean is tight; seeded, tolerances pinned from the run.
    """
    rng = np.random.default_rng(SEED + 3)
    finals2 = np.array([kt_histogram_martingale(rng.random(2), N_BINS)[-1] for _ in range(20_000)])
    assert abs(float(finals2.mean()) - 1.0) <= 0.05
    finals5 = np.array([kt_histogram_martingale(rng.random(5), N_BINS)[-1] for _ in range(20_000)])
    assert abs(float(finals5.mean()) - 1.0) <= 0.15
    rng2 = np.random.default_rng(SEED + 4)
    mix5 = np.array(
        [restart_mixture_kt_martingale(rng2.random(5), N_BINS)[-1] for _ in range(3_000)]
    )
    assert abs(float(mix5.mean()) - 1.0) <= 0.1  # Prop. 3.3: also mean-one


# ---------------------------------------------------------------------------
# (c) restart mixture: weights, tail term, component structure
# ---------------------------------------------------------------------------


def test_restart_weights_uniform_and_geometric() -> None:
    u = restart_weights(5, "uniform")
    np.testing.assert_allclose(u, np.full(5, 0.2))
    g = restart_weights(4, "geometric", gamma=0.5)
    np.testing.assert_allclose(g, np.array([1.0, 0.5, 0.25, 0.125]) / 1.875)
    assert g.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gamma"):
        restart_weights(4, "geometric")
    with pytest.raises(ValueError, match="gamma"):
        restart_weights(4, "geometric", gamma=1.5)
    with pytest.raises(ValueError, match="uniform"):
        restart_weights(4, "bogus")
    with pytest.raises(ValueError, match="horizon"):
        restart_weights(0)


def test_restart_mixture_m0_tail_and_single_component() -> None:
    """M_0 = 1 via the weight tail; a point mass reproduces one component."""
    rng = np.random.default_rng(SEED + 8)
    p = rng.random(30)
    mix = restart_mixture_kt_martingale(p, 10)
    assert mix[0] == 1.0
    tail = np.array([(30 - t) / 30 for t in range(31)])
    assert bool(np.all(mix >= tail - 1e-12))  # components are nonnegative
    # all mass on restart s0 = 5: for t > 5 the mixture IS the component,
    # i.e. the unrestarted KT process applied to p[5:] (their Eq. 4).
    w = np.zeros(30)
    w[5] = 1.0
    point = restart_mixture_kt_martingale(p, 10, weights=w)
    comp = kt_histogram_martingale(p[5:], 10)
    np.testing.assert_allclose(point[:6], np.ones(6), atol=1e-12)
    np.testing.assert_allclose(point[6:], comp[1:], rtol=1e-10)


def test_restart_mixture_matches_equation5_bruteforce() -> None:
    """Direct triple-loop implementation of Eq. (4)+(5) with uneven weights."""
    rng = np.random.default_rng(SEED + 9)
    p = rng.random(8)
    j = 3
    pi = np.arange(1.0, 9.0)
    pi /= pi.sum()
    bins = [min(int(x * j), j - 1) for x in p]
    brute = [1.0]
    for t in range(1, 9):
        total = 0.0
        for s in range(t):
            counts = [0] * j
            w = 1.0
            for i in range(s + 1, t + 1):
                b = bins[i - 1]
                w *= j * (counts[b] + 0.5) / ((i - 1 - s) + j / 2.0)
                counts[b] += 1
            total += pi[s] * w
        total += float(pi[t:].sum())
        brute.append(total)
    mix = restart_mixture_kt_martingale(p, j, weights=pi)
    np.testing.assert_allclose(mix, np.array(brute), rtol=1e-10)


def test_restart_mixture_fail_closed() -> None:
    p = np.full(10, 0.5)
    with pytest.raises(ValueError, match="len"):
        restart_mixture_kt_martingale(p, 5, weights=np.full(9, 1.0 / 9))
    with pytest.raises(ValueError, match="nonnegative"):
        restart_mixture_kt_martingale(p, 5, weights=np.arange(-1.0, 9.0))
    with pytest.raises(ValueError, match="positive"):
        restart_mixture_kt_martingale(p, 5, weights=np.zeros(10))
    with pytest.raises(ValueError, match="uniform"):
        restart_mixture_kt_martingale(p, 5, weights="bogus")
    with pytest.raises(ValueError, match="gamma"):
        restart_mixture_kt_martingale(p, 5, gamma=0.0)


# ---------------------------------------------------------------------------
# (d) crossing semantics and composition with the existing Ville modules
# ---------------------------------------------------------------------------


def test_crossing_time_strict_vs_nonstrict() -> None:
    path = np.array([1.0, 5.0, 10.0, 20.0])
    assert crossing_time(path, 10.0, strict=False) == 2
    assert crossing_time(path, 10.0, strict=True) == 3
    assert crossing_time(path, 20.0, strict=False) == 3
    assert crossing_time(path, 20.0, strict=True) is None
    with pytest.raises(ValueError, match="non-empty"):
        crossing_time(np.array([]), 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        crossing_time(np.array([1.0, -2.0]), 1.0)
    with pytest.raises(ValueError, match="threshold"):
        crossing_time(path, 0.0)


def test_ville_crossing_matches_existing_modules() -> None:
    """Non-strict crossing at 1/alpha == conformal_martingale.martingale_alarm.

    Composition-only check: this lane sharpens thresholds in a separate
    layer; the existing Ville alarms in conformal_martingale / e_detectors
    are untouched and remain the anytime-valid fallback.
    """
    rng = np.random.default_rng(SEED + 10)
    p = rng.random(200)
    path = kt_histogram_martingale(p, N_BINS)
    mine = crossing_time(path, VILLE, strict=False)
    theirs = martingale_alarm(path, ALPHA)["alarm_index"]
    assert mine == theirs
    cal = calibrate_conformal_martingale(ALPHA, horizon=50, n_reference=99, seed=SEED + 11)
    assert cal.ville_threshold == alarm_threshold(ALPHA) == VILLE


# ---------------------------------------------------------------------------
# (e) conformal specialization: reference banks, calibration, fail-closed
# ---------------------------------------------------------------------------


def test_conformal_null_path_maxima_shape_and_determinism() -> None:
    m1 = conformal_null_path_maxima(40, 60, construction="kt", n_bins=10, seed=SEED + 12)
    m2 = conformal_null_path_maxima(40, 60, construction="kt", n_bins=10, seed=SEED + 12)
    np.testing.assert_array_equal(m1, m2)  # determinism pinned
    assert m1.shape == (40,)
    assert bool(np.all(m1 >= 1.0))  # path maxima dominate M_0 = 1
    m3 = conformal_null_path_maxima(40, 60, construction="kt", n_bins=10, seed=SEED + 13)
    assert not np.array_equal(m1, m3)  # different seed -> different bank


def test_conformal_calibration_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_reference"):
        conformal_null_path_maxima(0, 50)
    with pytest.raises(ValueError, match="horizon"):
        conformal_null_path_maxima(10, 0)
    with pytest.raises(ValueError, match="construction"):
        conformal_null_path_maxima(10, 50, construction="bogus")
    with pytest.raises(ValueError, match="n_bins"):
        conformal_null_path_maxima(10, 50, n_bins=1)
    with pytest.raises(ValueError, match="too few"):
        calibrate_conformal_martingale(ALPHA, horizon=50, n_reference=5, seed=SEED + 14)


def test_calibrate_conformal_martingale_fields() -> None:
    cal = calibrate_conformal_martingale(
        ALPHA,
        horizon=100,
        construction="restart_mixture",
        n_bins=10,
        n_reference=99,
        seed=SEED + 15,
    )
    assert cal.method == "reference-null:conformal-restart_mixture"
    assert cal.k_order == 95  # ceil(0.95 * 100)
    assert cal.n_reference == 99
    assert np.isfinite(cal.threshold) and cal.threshold >= 1.0


# ---------------------------------------------------------------------------
# (f) Monte-Carlo validation on seeded SYNTHETIC streams (paper Sec. 5 design)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("construction", "n_banks", "bank_size", "n_eval"),
    [
        ("kt", N_BANKS_KT, BANK_KT, N_EVAL_KT),
        ("restart_mixture", N_BANKS_RKT, BANK_RKT, N_EVAL_RKT),
    ],
)
def test_type_i_control_ville_and_calibrated(
    construction: str, n_banks: int, bank_size: int, n_eval: int
) -> None:
    """Empirical finite-horizon FA <= alpha + tol for BOTH boundaries (Thm 2.2).

    SYNTHETIC exchangeable null: iid N(0,1) score streams -> conformal
    p-values -> the SAME e-process path for the calibrated (strict) and the
    Ville (non-strict) rule — the paper's paired C-vs-V design isolates the
    boundary effect. Fresh null paths are shared across independently drawn
    calibration banks; the bank-averaged calibrated FA estimates the
    unconditional type-I error, which the exchangeability guarantee bounds by
    alpha (never anti-conservative on average), while Ville stays valid too.
    """
    paths = _eval_paths(construction, n_eval, T_FA)
    ville_hits = np.array(
        [crossing_time(r, VILLE, strict=False) is not None for r in paths], dtype=bool
    )
    fa_ville = float(ville_hits.mean())
    assert fa_ville <= ALPHA + FA_TOL

    fa_per_bank = []
    for b in range(n_banks):
        cal = calibrate_conformal_martingale(
            ALPHA,
            horizon=T_FA,
            construction=construction,
            n_bins=N_BINS,
            n_reference=bank_size,
            seed=SEED + BANK_OFFSET + 1000 * b,
        )
        hits = np.array(
            [crossing_time(r, cal.threshold, strict=True) is not None for r in paths], dtype=bool
        )
        fa_per_bank.append(float(hits.mean()))
    fa_mean = float(np.mean(fa_per_bank))
    assert fa_mean <= ALPHA + FA_TOL, f"bank-averaged calibrated FA {fa_mean} exceeds alpha+tol"
    assert max(fa_per_bank) <= ALPHA + BANK_TOL, f"per-bank FA {max(fa_per_bank)} anti-conservative"


def test_calibrated_boundary_is_sharper_on_average() -> None:
    """Median calibrated threshold below the Ville boundary (boundary reduction).

    Paper Table 1 reports median threshold/Ville ratios ~0.64-0.72 for KT/RKT
    at T=2000; a single bank CAN calibrate above 1/alpha (the paper does not
    truncate, and validity never relies on sharpness < 1), so the reduction
    is asserted on the median across banks at T=250.
    """
    sharps = [
        calibrate_conformal_martingale(
            ALPHA,
            horizon=T_FA,
            construction="kt",
            n_bins=N_BINS,
            n_reference=BANK_KT,
            seed=SEED + BANK_OFFSET + 1000 * b,
        ).sharpness
        for b in range(N_BANKS_KT)
    ]
    assert float(np.median(sharps)) < 1.0
    assert min(sharps) > 0.0


def test_calibrated_detects_earlier_under_planted_shift() -> None:
    """Paired delay gain under the paper's change-point model (their Eq. 7).

    SYNTHETIC: X_1..X_tau iid N(0,1), then iid N(mu,1), tau = T/2, mu = 1.0,
    restart-mixture KT construction. The SAME e-process paths are thresholded
    at the calibrated boundary (strict) and at Ville (non-strict): when
    c_hat < 1/alpha the calibrated crossing is pathwise never later, and the
    restricted mean detection delay (their Sec. 5.1, non-detections censored
    at T - tau + 1) must improve on average.
    """
    cal = calibrate_conformal_martingale(
        ALPHA,
        horizon=T_DELAY,
        construction="restart_mixture",
        n_bins=N_BINS,
        n_reference=BANK_DELAY,
        seed=DELAY_BANK_SEED,
    )
    assert cal.threshold < cal.ville_threshold  # sharper boundary on this bank

    paths = []
    for rep in range(N_REPS_DELAY):
        rng = np.random.default_rng(SEED + SHIFT_SCORE_OFFSET + rep)
        scores = np.concatenate(
            [rng.standard_normal(TAU_DELAY), rng.standard_normal(T_DELAY - TAU_DELAY) + MU_DELAY]
        )
        p = conformal_p_values(scores, seed=SEED + SHIFT_RANDOMIZER_OFFSET + rep)
        paths.append(restart_mixture_kt_martingale(p, N_BINS))
    arr = np.stack(paths)

    sum_c = detection_delay_summary(
        arr, threshold=cal.threshold, change_time=TAU_DELAY, strict=True
    )
    sum_v = detection_delay_summary(arr, threshold=VILLE, change_time=TAU_DELAY, strict=False)
    d_c = np.asarray(sum_c["detection_times"], dtype=np.int64)
    d_v = np.asarray(sum_v["detection_times"], dtype=np.int64)
    inf_c = np.where(d_c >= 0, d_c, np.iinfo(np.int64).max)
    inf_v = np.where(d_v >= 0, d_v, np.iinfo(np.int64).max)
    assert bool(np.all(inf_c <= inf_v))  # pathwise dominance when c_hat < 1/alpha
    rmdd_c = float(sum_c["rmdd"])
    rmdd_v = float(sum_v["rmdd"])
    assert rmdd_c <= rmdd_v
    gain = rmdd_v - rmdd_c
    assert gain > 0.5, f"mean detection-delay gain {gain} too small"  # observed ~5 observations
    assert float(sum_c["conditional_power"]) >= 0.9  # saturated-power regime (their Sec. 5.3)
    assert float(sum_v["conditional_power"]) >= 0.9


def test_detection_delay_summary_handcrafted() -> None:
    """Exact Sec. 5.1 bookkeeping: pre-change FA, conditional power, censoring."""
    t_len = 11  # T = 10, tau = 5 -> censoring value T - tau + 1 = 6
    paths = np.ones((4, t_len))
    paths[0, 4] = 25.0  # pre-change crossing at index 4 <= tau -> false alarm
    paths[1, 7] = 25.0  # detection at 7, delay 2
    paths[2, :] = np.maximum(paths[2, :], 5.0)  # never crosses 10 -> censored 6
    paths[3, 9] = 12.0  # detection at 9 (non-strict, thr 10), delay 4
    out = detection_delay_summary(paths, threshold=10.0, change_time=5, strict=False)
    np.testing.assert_array_equal(out["detection_times"], np.array([4, 7, -1, 9]))
    assert out["fa_rate_prechange"] == pytest.approx(0.25)
    assert out["conditional_power"] == pytest.approx(2.0 / 3.0)
    assert out["rmdd"] == pytest.approx((2.0 + 6.0 + 4.0) / 3.0)
    assert out["n_reps"] == 4
    # strict rule: the exact hit 12.0 at index 9 no longer crosses threshold 12
    strict = detection_delay_summary(paths, threshold=12.0, change_time=5, strict=True)
    np.testing.assert_array_equal(strict["detection_times"], np.array([4, 7, -1, -1]))
    assert strict["conditional_power"] == pytest.approx(1.0 / 3.0)
    assert strict["rmdd"] == pytest.approx((2.0 + 6.0 + 6.0) / 3.0)
    with pytest.raises(ValueError, match="change_time"):
        detection_delay_summary(paths, threshold=10.0, change_time=0)
    with pytest.raises(ValueError, match="change_time"):
        detection_delay_summary(paths, threshold=10.0, change_time=10)
    with pytest.raises(ValueError, match="2-D"):
        detection_delay_summary(np.ones(11), threshold=10.0, change_time=5)


def test_end_to_end_determinism_pinned() -> None:
    """Same seeds -> bit-identical calibration, paths, and summaries."""
    cal_a = calibrate_conformal_martingale(
        ALPHA, horizon=80, construction="kt", n_bins=N_BINS, n_reference=99, seed=SEED + 16
    )
    cal_b = calibrate_conformal_martingale(
        ALPHA, horizon=80, construction="kt", n_bins=N_BINS, n_reference=99, seed=SEED + 16
    )
    assert cal_a == cal_b
    streams = _null_p_streams(5, 80)
    pa = np.stack([kt_histogram_martingale(p, N_BINS) for p in streams])
    pb = np.stack([kt_histogram_martingale(p, N_BINS) for p in streams])
    np.testing.assert_array_equal(pa, pb)
    sa = detection_delay_summary(pa, threshold=cal_a.threshold, change_time=40, strict=True)
    sb = detection_delay_summary(pb, threshold=cal_b.threshold, change_time=40, strict=True)
    np.testing.assert_array_equal(sa["detection_times"], sb["detection_times"])
    assert sa["rmdd"] == sb["rmdd"]
