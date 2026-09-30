"""Tests for conformal_e_detectors: minimax-optimal conformal change detection.

Constructions under test (Bhattacharyya & Ramdas 2026, arXiv:2609.27179):
restarted conformal mixture martingales aggregated with deterministic restart
weights — conformal e-processes (summable weights, PFA <= alpha at ANY
stopping time, their Thm. 2.7(1)/Eq. (17)) and conformal e-detectors (unit
weights, ARL >= gamma plus the linear early-alarm bound m/gamma, their
Thm. 2.7(2)/Eq. (19)-(20), null-bet modification Prop. 4.10) — matched
against the existing repo constructions: Vovk (2021) conformal test
martingales (conformal_martingale power/mixture/jumper) and the parametric
mixture-SR e-detector (e_detectors.EDetectorGaussian).

All streams are seeded SYNTHETIC draws (np.random.default_rng, pinned seeds —
determinism, no market data, no market evidence). The delay-comparison tests
ILLUSTRATE the paper's asymptotic rates at finite samples — existing methods
Omega(T) / Omega(sqrt(ARL)) (their Examples 2.2-2.5) vs optimal Theta(log T) /
Theta(log ARL) (their Cor. 3.6/3.8, restricted minimax Thm. 3.9/3.10) — they
do not PROVE them. Monte-Carlo tolerances follow the repo convention
(test_e_detectors / test_confidence_sequences): empirical false-alarm rate
<= alpha + 0.02 at >= 50 replicates; the repo comparator mixture martingale
gets the looser +0.10 slack at 16 replicates (its Ville bound is <= alpha in
expectation; small-sample binomial noise). E-process means are checked
against <= 1 + 0.25 (optional stopping, heavy-tailed martingale).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal_e_detectors import (
    ARL_METHODS,
    CTM_THETA,
    PFA_METHODS,
    ConformalDetectionResult,
    ConformalRestartDetector,
    DelayStudyRow,
    PowerBetting,
    arl_calibration_study,
    arl_delay_study,
    cmm_log_wealth,
    conformal_e_detector,
    conformal_e_process,
    ctm_log_wealth,
    delay_growth_study,
    false_alarm_study,
    first_crossing,
    first_crossing_log,
    pfa_delay_study,
    power_betting_class,
    power_log_bets,
    restart_weights,
    summarize_delays,
    synthetic_changepoint_scores,
    vovk_cusum_stat,
    vovk_sr_stat,
)
from quant_fund.metrics.conformal_martingale import conformal_p_values

ALPHA = 0.05
FA_TOL = 0.02  # binomial slack convention (test_e_detectors.py)
EPROC_TOL = 0.25  # optional-stopping mean slack (heavy-tailed martingale)


def _mild_class() -> PowerBetting:
    """Light-tailed betting class for martingale-mean checks (E f^2 < inf)."""
    return power_betting_class(thetas=(0.75, 1.0), sides="both")


# ---------------------------------------------------------------------------
# (a) betting class and restart-weight math
# ---------------------------------------------------------------------------


def test_power_densities_integrate_to_one() -> None:
    """One-step fairness: int_0^1 f_theta = 1 for every atom, both sides."""
    from scipy.integrate import quad

    bet = power_betting_class()  # default two-sided theta grid
    for k in range(bet.n_atoms):
        theta = float(bet.thetas[k])
        val_l, err_l = quad(
            lambda u, th=theta: float(np.exp(power_log_bets(u, th, "left"))[0]), 0.0, 1.0
        )
        assert val_l == pytest.approx(1.0, abs=1e-8 + err_l)
        val_r, err_r = quad(
            lambda u, th=theta: float(np.exp(power_log_bets(u, th, "right"))[0]), 0.0, 1.0
        )
        assert val_r == pytest.approx(1.0, abs=1e-8 + err_r)


def test_log_bets_shapes_mirror_and_null_bet() -> None:
    bet = power_betting_class(thetas=(0.15, 1.0), sides="both")
    assert bet.n_atoms == 4
    scalar = bet.log_bets(0.3)
    assert scalar.shape == (4,)
    path = bet.log_bets(np.array([0.1, 0.5, 0.9]))
    assert path.shape == (3, 4)
    # theta = 1 atoms are the null bet f == 1 (log f == 0), both sides
    assert float(scalar[1]) == 0.0 and float(scalar[3]) == 0.0
    # left atom at p == right atom at 1 - p (mirror symmetry)
    left = power_betting_class(thetas=(0.15,), sides="left")
    right = power_betting_class(thetas=(0.15,), sides="right")
    assert float(left.log_bets(0.2)[0]) == pytest.approx(float(right.log_bets(0.8)[0]))
    # closed form: f_0.15(p) = 0.15 p^{-0.85} (the paper's Fig. 1 density)
    assert float(left.log_bets(0.2)[0]) == pytest.approx(np.log(0.15 * 0.2**-0.85))


def test_restart_weights_normalization_and_shape() -> None:
    for kind in ("polynomial", "near_harmonic"):
        w = restart_weights(kind, 500)
        assert w.shape == (500,)
        assert float(np.sum(w)) == pytest.approx(1.0)
        assert bool(np.all(w > 0.0)) and bool(np.all(np.diff(w) < 0.0))
    w_poly = restart_weights("polynomial", 500, eta=0.10)
    assert float(w_poly[1] / w_poly[0]) == pytest.approx(2.0**-1.1)
    w_nh = restart_weights("near_harmonic", 500)
    assert float(w_nh[1] / w_nh[0]) == pytest.approx(
        (np.log(np.e + 1.0) / np.log(np.e + 2.0)) ** 2 / 2.0
    )
    w_unit = restart_weights("unit", 10)
    assert bool(np.all(w_unit == 1.0))


# ---------------------------------------------------------------------------
# (b) engine correctness against the definition S_t = sum_k w_k M_{k,t}
# ---------------------------------------------------------------------------


def test_sum_and_max_match_hand_computed_single_atom() -> None:
    """Single atom theta=0.5, p=(0.5,0.5,0.5) => f = sqrt(0.5); weights sum to 1."""
    bet = power_betting_class(thetas=(0.5,), sides="left")
    w = np.array([0.5, 0.3, 0.2])
    f = 0.5**0.5
    s_expected = [0.5 * f, 0.5 * f**2 + 0.3 * f, 0.5 * f**3 + 0.3 * f**2 + 0.2 * f]
    z_expected = [0.5 * f, max(0.5 * f**2, 0.3 * f), max(0.5 * f**3, 0.3 * f**2, 0.2 * f)]
    det_s = conformal_e_process(alpha=ALPHA, n_max=3, weights=w, betting=bet, stat="sum")
    res_s = det_s.run_p_values([0.5, 0.5, 0.5], stop_on_alarm=False)
    np.testing.assert_allclose(res_s.statistic, s_expected, rtol=1e-12)
    # e-process completion: S~_t = S_t + sum_{k>t} w_k (their Eq. (6))
    assert res_s.e_process is not None
    np.testing.assert_allclose(res_s.e_process, np.array(s_expected) + np.array([0.5, 0.2, 0.0]))
    det_z = conformal_e_process(alpha=ALPHA, n_max=3, weights=w, betting=bet, stat="max")
    res_z = det_z.run_p_values([0.5, 0.5, 0.5], stop_on_alarm=False)
    np.testing.assert_allclose(res_z.statistic, z_expected, rtol=1e-12)
    assert res_z.e_process is None


def test_statistics_match_bruteforce_restart_definition() -> None:
    """Engine recursions == the O(T^2 K) definition sum_k w_k int prod f_theta dPi."""
    rng = np.random.default_rng(7)
    bet = power_betting_class(thetas=(0.15, 0.5, 1.0), sides="both")
    n_max = 8
    w = restart_weights("polynomial", n_max, eta=0.1)
    p = rng.uniform(size=n_max)
    log_f = bet.log_bets(p)  # (T, K)
    prior = np.exp(bet.log_prior)
    brute = {"sum": np.empty(n_max), "max": np.empty(n_max)}
    for t in range(n_max):
        ms = [float(np.sum(prior * np.exp(log_f[k : t + 1].sum(axis=0)))) for k in range(t + 1)]
        arr = np.asarray(ms) * w[: t + 1]
        brute["sum"][t] = float(arr.sum())
        brute["max"][t] = float(arr.max())
    for stat in ("sum", "max"):
        det = conformal_e_process(alpha=ALPHA, n_max=n_max, weights=w, betting=bet, stat=stat)  # type: ignore[arg-type]
        res = det.run_p_values(p, stop_on_alarm=False)
        np.testing.assert_allclose(res.statistic, brute[stat], rtol=1e-9)


def test_e_process_mean_one_under_uniform_p_values() -> None:
    """S~ is a martingale with initial value ||w||_1 = 1: E[S~_U] = 1 (MC)."""
    rng = np.random.default_rng(77)
    ends = []
    for _ in range(200):
        det = conformal_e_process(
            alpha=ALPHA, n_max=150, weights="polynomial", betting=_mild_class(), stat="sum"
        )
        res = det.run_p_values(rng.uniform(size=150), stop_on_alarm=False)
        assert res.e_process is not None
        ends.append(float(res.e_process[-1]))
    mean = float(np.mean(ends))
    assert 0.4 <= mean <= 1.0 + EPROC_TOL


def test_sequential_update_matches_batch_and_repo_p_values() -> None:
    scores = synthetic_changepoint_scores(60, 40, shift=1.5, seed=3)
    d1 = conformal_e_process(alpha=ALPHA, n_max=100, stat="sum", seed=11)
    r1 = d1.run(scores, stop_on_alarm=False)
    d2 = conformal_e_process(alpha=ALPHA, n_max=100, stat="sum", seed=11)
    vals = [d2.update(float(s)) for s in scores]
    np.testing.assert_allclose(r1.statistic, np.asarray(vals), rtol=0.0, atol=1e-12)
    # identical p-values to the repo's Vovk-2021 sequential conformal p-values
    np.testing.assert_array_equal(r1.p_values, conformal_p_values(scores, seed=11))


def test_run_stop_on_alarm_truncates_paths() -> None:
    """Extreme p (1e-8) alarms at the first step; paths truncate there."""
    bet = power_betting_class(thetas=(0.15,), sides="left")
    det = conformal_e_process(alpha=ALPHA, n_max=5, betting=bet, stat="max")
    res = det.run_p_values([1e-8] * 5, stop_on_alarm=True)
    assert res.alarm_index == 0
    assert res.statistic.shape == (1,) and res.p_values.shape == (1,)
    det2 = conformal_e_process(alpha=ALPHA, n_max=5, betting=bet, stat="max")
    res2 = det2.run_p_values([1e-8] * 5, stop_on_alarm=False)
    assert res2.alarm_index == 0 and res2.statistic.shape == (5,)


def test_determinism_same_seed_identical_paths() -> None:
    scores = synthetic_changepoint_scores(200, 100, shift=1.2, seed=5)
    outs = []
    for _ in range(2):
        det = conformal_e_process(alpha=ALPHA, n_max=300, stat="max", seed=21)
        outs.append(det.run(scores))
    np.testing.assert_array_equal(outs[0].statistic, outs[1].statistic)
    assert outs[0].alarm_index == outs[1].alarm_index
    # study-level determinism: identical rows from identical seeds
    rows_a = pfa_delay_study(150, n_reps=6, seed=99, methods=("optimal_sum", "cmm_grid"))
    rows_b = pfa_delay_study(150, n_reps=6, seed=99, methods=("optimal_sum", "cmm_grid"))
    assert rows_a == rows_b


def test_pfa_and_arl_bound_properties() -> None:
    det = conformal_e_process(alpha=ALPHA, n_max=100)
    assert det.pfa_bound == pytest.approx(ALPHA)  # ||w||_1 = 1, b = 1/alpha
    # arl_bound = b / ||w||_inf = 1/(alpha * w_1) with w_1 the largest weight
    w = restart_weights("polynomial", 100)
    assert det.arl_bound == pytest.approx(1.0 / (ALPHA * float(w[0])))
    w_half = restart_weights("polynomial", 50) * 0.5
    det_h = conformal_e_process(alpha=ALPHA, n_max=50, weights=w_half, stat="sum")
    assert det_h.pfa_bound == pytest.approx(0.5 * ALPHA)
    edet = conformal_e_detector(gamma=100.0, n_max=500)
    assert edet.arl_bound == pytest.approx(100.0)  # unit weights
    assert edet.pfa_bound == 1.0  # honest: unit weights give NO PFA control
    assert isinstance(det, ConformalRestartDetector)


def test_sure_stopping_with_null_bet() -> None:
    """Prop. 4.10: S-bar_t >= rho * W_t = rho * t forces a stop by ceil(b/rho)."""
    gamma, rho = 50.0, 0.1
    det = conformal_e_detector(gamma=gamma, n_max=500, rho=rho, stat="sum")
    rng = np.random.default_rng(4)
    res = det.run_p_values(rng.uniform(size=500), stop_on_alarm=True)
    bound = int(np.ceil(gamma / rho))
    assert res.alarm_index is not None
    assert res.alarm_index + 1 <= bound
    assert float(res.statistic[-1]) >= gamma


def test_vovk_helper_math() -> None:
    """Vovk SR/CUSUM statistics against their definitional sums (Eq. (10))."""
    rng = np.random.default_rng(8)
    lb = rng.normal(size=6)
    sr = vovk_sr_stat(lb)
    assert sr[0] == 0.0
    for n in range(2, 7):  # 1-based
        brute = sum(float(np.exp(np.sum(lb[i:n]))) for i in range(1, n))
        assert sr[n - 1] == pytest.approx(brute, rel=1e-10)
    zeros = np.zeros(5)
    cus = vovk_cusum_stat(zeros)  # S_n/S_i == 1 everywhere
    np.testing.assert_allclose(cus, np.ones(5))
    lw = np.array([0.0, -1.0, -2.0, 0.5, 1.0])
    cus2 = vovk_cusum_stat(lw)
    assert bool(np.all(cus2 >= 1.0))
    assert cus2[4] == pytest.approx(float(np.exp(1.0 - (-2.0))))  # min log S_i = -2
    assert first_crossing([1.0, 20.0, 5.0], 20.0) == 1
    assert first_crossing([1.0, 2.0], 20.0) is None
    assert first_crossing_log([0.0, np.log(20.0)], 20.0) == 1
    # CTM/CMM log wealth paths are consistent
    p = rng.uniform(size=50)
    np.testing.assert_allclose(
        ctm_log_wealth(p, CTM_THETA), np.cumsum(power_log_bets(p, CTM_THETA))
    )
    bet = power_betting_class(thetas=(0.15,), sides="left")
    np.testing.assert_allclose(
        cmm_log_wealth(p, bet), ctm_log_wealth(p, CTM_THETA), rtol=1e-12
    )  # single-atom mixture == CTM


def test_summarize_delays_mechanics() -> None:
    row = summarize_delays("m", "pfa", [None, 9, 12, 14, 3], changepoint=10, cap=5)
    assert isinstance(row, DelayStudyRow)
    assert row.false_alarm_rate == pytest.approx(0.4)  # indices 9 and 3 are <= T
    assert row.detection_rate == pytest.approx(2.0 / 3.0)  # of 3 clean reps, 2 alarmed
    assert row.median_capped_delay == pytest.approx(5.0)  # delays [5, 3, 5]
    assert row.mean_capped_delay == pytest.approx(13.0 / 3.0)
    assert row.median_delay_over_log_t == pytest.approx(5.0 / np.log(10))
    with pytest.raises(ValueError):
        summarize_delays("m", "pfa", [1, 2], changepoint=10, cap=5)  # all false-alarmed
    with pytest.raises(ValueError):
        summarize_delays("m", "pfa", [15], changepoint=10, cap=5)  # index past horizon
    with pytest.raises(ValueError):
        summarize_delays("m", "bogus", [None], changepoint=10, cap=5)
    with pytest.raises(ValueError):
        summarize_delays("m", "pfa", [], changepoint=10, cap=5)


# ---------------------------------------------------------------------------
# (c) Monte-Carlo time-uniform type-I control under exchangeable nulls
# ---------------------------------------------------------------------------


def test_false_alarm_control_gaussian_null_all_methods() -> None:
    """P(ever cross 1/alpha | iid N(0,1), ANY stopping) <= alpha + tol.

    The ever-crossing event over the horizon dominates every crossing-based
    stopping rule, so this is the time-uniform (any-stopping) guarantee.
    """
    rates = false_alarm_study(regime="pfa", n_reps=60, horizon=600, alpha=ALPHA, seed=44)
    assert set(rates) == set(PFA_METHODS) - {"repo_jumper"}
    for m, r in rates.items():
        slack = 0.10 if m == "repo_power_mixture" else FA_TOL
        assert r <= ALPHA + slack, (m, r)


@pytest.mark.parametrize("null", ["uniform", "ties"])
def test_false_alarm_control_other_exchangeable_nulls(null: str) -> None:
    """Validity is distribution-free: U(0,1) and heavily-tied discrete nulls."""
    rates = false_alarm_study(
        regime="pfa",
        n_reps=50,
        horizon=400,
        alpha=ALPHA,
        null=null,  # type: ignore[arg-type]
        seed=45,
        methods=("vovk_ctm", "cmm_grid", "optimal_sum", "optimal_max"),
    )
    for m, r in rates.items():
        assert r <= ALPHA + FA_TOL, (m, r)


def test_e_process_optional_stopping_means() -> None:
    """E[S~_tau] <= 1 at (a) independent-uniform and (b) crossing-or-horizon taus."""
    rng = np.random.default_rng(123)
    uni_vals, cross_vals = [], []
    for _ in range(200):
        det = conformal_e_process(
            alpha=ALPHA, n_max=150, weights="polynomial", betting=_mild_class(), stat="sum"
        )
        res = det.run_p_values(rng.uniform(size=150), stop_on_alarm=False)
        assert res.e_process is not None
        e = res.e_process
        uni_vals.append(float(e[int(rng.integers(0, 150))]))
        hits = np.flatnonzero(e >= 3.0)
        cross_vals.append(float(e[int(hits[0])] if hits.size else e[-1]))
    assert float(np.mean(uni_vals)) <= 1.0 + EPROC_TOL
    assert float(np.mean(cross_vals)) <= 1.0 + EPROC_TOL


def test_arl_calibration_empirical() -> None:
    """ARL >= gamma, sure stop by ceil(gamma/rho), P(tau <= m) <= m/gamma (Eq. 19)."""
    cal = arl_calibration_study(gamma=100.0, rho=0.1, n_reps=60, seed=47)
    assert float(cal["empirical_arl"]) >= float(cal["gamma"])  # 159.7 observed
    assert int(cal["max_tau"]) <= int(cal["sure_stop_bound"])  # 332 <= 1000
    for m, rate, bound in cal["early_alarms"]:  # type: ignore[union-attr]
        assert float(rate) <= float(bound) + FA_TOL, (m, rate, bound)


# ---------------------------------------------------------------------------
# (d) matched-delay comparisons on seeded SYNTHETIC planted-changepoint streams
# ---------------------------------------------------------------------------


def test_pfa_matched_delay_optimal_beats_existing() -> None:
    """At matched threshold 1/alpha: restarted e-processes detect far faster.

    T=400, N(0,1)->N(1.2,1), cap 5T, 16 seeded replicates (the paper's Sec. 6
    protocol). Finite-sample demonstration, not a proof of the rates.
    """
    rows = {r.method: r for r in pfa_delay_study(400, n_reps=16, seed=42)}
    assert set(rows) == set(PFA_METHODS)
    opt_s, opt_m = rows["optimal_sum"], rows["optimal_max"]
    for m in ("vovk_ctm", "cmm_grid", "optimal_sum", "optimal_max", "repo_jumper"):
        assert rows[m].false_alarm_rate <= ALPHA + FA_TOL, m
    # repo mixture at 16 reps: Ville bound is alpha in expectation; binomial noise
    assert rows["repo_power_mixture"].false_alarm_rate <= ALPHA + 0.10
    assert opt_s.detection_rate >= 0.9 and opt_m.detection_rate >= 0.9
    assert opt_s.median_delay_over_log_t <= 6.0 and opt_m.median_delay_over_log_t <= 6.0
    assert opt_m.median_capped_delay >= opt_s.median_capped_delay  # Z_t <= S_t pathwise
    assert rows["cmm_grid"].median_capped_delay >= 2.0 * opt_s.median_capped_delay
    assert rows["repo_power_mixture"].median_capped_delay >= opt_s.median_capped_delay
    assert rows["repo_jumper"].median_capped_delay >= opt_s.median_capped_delay
    # Vovk CTM must first repay the pre-change dilution: pinned at the cap here
    assert rows["vovk_ctm"].detection_rate <= 0.2
    assert rows["vovk_ctm"].median_capped_delay >= 10.0 * opt_s.median_capped_delay


def test_delay_growth_polynomial_vs_log_gap() -> None:
    """Normalized median delay / log T: flat for optimal, diverging for existing.

    Mirrors the paper's Fig. 5 (Vovk CTM) and Fig. 6 (CMM): their delays are
    Omega(T)-scale while the restarted statistics stay Theta(log T)-scale.
    Finite-sample Monte-Carlo evidence — ILLUSTRATES, does not prove, rates.
    """
    rows = delay_growth_study((200, 400, 800), n_reps=16, seed=43)
    by = {(r.changepoint, r.method): r for r in rows}
    for method in ("optimal_sum", "optimal_max"):
        norm = [by[(t, method)].median_delay_over_log_t for t in (200, 400, 800)]
        assert max(norm) <= 6.0, (method, norm)  # Theta(log T): normalized delay flat
        assert max(norm) - min(norm) <= 2.0, (method, norm)
    cmm_norm = [by[(t, "cmm_grid")].median_delay_over_log_t for t in (200, 400, 800)]
    assert cmm_norm[-1] >= 1.5 * cmm_norm[0]  # CMM dilution: normalized delay diverges
    opt_growth = (
        by[(800, "optimal_sum")].median_capped_delay / by[(200, "optimal_sum")].median_capped_delay
    )
    cmm_growth = (
        by[(800, "cmm_grid")].median_capped_delay / by[(200, "cmm_grid")].median_capped_delay
    )
    assert cmm_growth >= 1.8 * opt_growth
    for t in (200, 400, 800):
        ctm = by[(t, "vovk_ctm")]
        assert ctm.detection_rate == 0.0  # never beats the 5T cap at these scales
        assert ctm.median_capped_delay == pytest.approx(5.0 * t)  # cap == Omega(T)
        assert ctm.median_delay_over_log_t >= 100.0
    assert (
        by[(800, "vovk_ctm")].median_capped_delay
        >= 50.0 * by[(800, "optimal_sum")].median_capped_delay
    )
    assert (
        by[(800, "cmm_grid")].median_capped_delay
        >= 3.0 * by[(800, "optimal_sum")].median_capped_delay
    )


def test_arl_matched_delay_optimal_beats_existing() -> None:
    """At matched ARL threshold gamma=1e4 (the paper's Fig. 2 protocol).

    Vovk CMM-SR/CUSUM (Eq. (13) applied to the mixture path) carry the
    pre-change dilution into every ratio; the restarted e-detectors mix
    freshly at every candidate changepoint. The favorable single-density
    vovk_sr_ctm is reported honestly: a bet matched to the alternative is
    not catastrophic at this scale — the Omega(sqrt(gamma)) worst case of
    their Example 2.3 is over unmatched bets. repo_mixture_sr is a
    PARAMETRIC reference (knows sigma=1 and the shift direction), not a
    distribution-free competitor.
    """
    rows = {r.method: r for r in arl_delay_study(400, gamma=1e4, cap=1500, n_reps=25, seed=46)}
    assert set(rows) == set(ARL_METHODS)
    opt_s = rows["optimal_sum"]
    for r in rows.values():
        # linear early-alarm bound at the changepoint: P(tau <= T) <= T/gamma = 0.04
        assert r.false_alarm_rate <= 400.0 / 1e4 + 0.06, r.method
    assert opt_s.detection_rate >= 0.9 and rows["optimal_max"].detection_rate >= 0.9
    assert opt_s.median_capped_delay <= 60.0
    assert rows["optimal_max"].median_capped_delay >= opt_s.median_capped_delay
    assert rows["vovk_sr_cmm"].median_capped_delay >= 2.0 * opt_s.median_capped_delay
    assert rows["vovk_cusum_cmm"].median_capped_delay >= 2.0 * opt_s.median_capped_delay
    assert rows["vovk_sr_ctm"].median_capped_delay >= opt_s.median_capped_delay
    assert rows["vovk_cusum_ctm"].detection_rate >= 0.4
    assert rows["repo_mixture_sr"].detection_rate >= 0.9


def test_left_shift_detected_without_direction_knowledge() -> None:
    """The two-sided grid adapts to a NEGATIVE mean shift (unknown direction)."""
    scores = synthetic_changepoint_scores(300, 300, shift=-1.2, seed=9)
    p = conformal_p_values(scores, seed=10)
    det = conformal_e_process(alpha=ALPHA, n_max=600, stat="max", seed=10)
    res = det.run_p_values(p)
    assert isinstance(res, ConformalDetectionResult)
    assert res.alarm_index is not None
    assert res.alarm_index >= 300  # no false alarm before the change
    assert res.alarm_index + 1 - 300 <= 150  # detected well within the horizon


def test_near_harmonic_weights_detect() -> None:
    """Rem. 3.7 weights are a valid PFA-regime choice and detect promptly."""
    scores = synthetic_changepoint_scores(400, 400, shift=1.2, seed=5)
    p = conformal_p_values(scores, seed=6)
    det = conformal_e_process(alpha=ALPHA, n_max=800, weights="near_harmonic", stat="max", seed=6)
    res = det.run_p_values(p)
    assert res.alarm_index is not None
    assert res.alarm_index + 1 - 400 <= 100


# ---------------------------------------------------------------------------
# (e) fail-closed edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: conformal_e_process(alpha=0.0, n_max=10),
        lambda: conformal_e_process(alpha=1.0, n_max=10),
        lambda: conformal_e_process(alpha=np.nan, n_max=10),
        lambda: conformal_e_process(alpha=-0.1, n_max=10),
        lambda: conformal_e_process(alpha=ALPHA, n_max=0),
        lambda: conformal_e_process(alpha=ALPHA, n_max=20_001),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights="unit"),  # ||w||_1 > 1
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights="bogus"),  # type: ignore[arg-type]
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights=np.ones(10) * 0.5),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights=np.ones(9)),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights=-np.ones(10)),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, weights=np.full(10, np.nan)),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, eta=0.0),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, eta=-1.0),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, stat="bogus"),  # type: ignore[arg-type]
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, rho=1.0),
        lambda: conformal_e_process(alpha=ALPHA, n_max=10, rho=-0.1),
        lambda: conformal_e_detector(gamma=0.5, n_max=10),
        lambda: conformal_e_detector(gamma=np.nan, n_max=10),
        lambda: conformal_e_detector(gamma=100.0, n_max=0),
        lambda: conformal_e_detector(gamma=100.0, n_max=10, weights=np.full(10, 2.0)),
        lambda: conformal_e_detector(gamma=100.0, n_max=10, weights="bogus"),  # type: ignore[arg-type]
        lambda: conformal_e_detector(gamma=100.0, n_max=10, stat="max", rho=0.05),
        lambda: power_betting_class(thetas=()),
        lambda: power_betting_class(thetas=(0.0,)),
        lambda: power_betting_class(thetas=(1.5,)),
        lambda: power_betting_class(thetas=(np.nan,)),
        lambda: power_betting_class(sides="bogus"),  # type: ignore[arg-type]
        lambda: restart_weights("bogus", 10),
        lambda: restart_weights("polynomial", 10, eta=0.0),
        lambda: restart_weights("polynomial", 0),
        lambda: restart_weights("unit", 20_001),
        lambda: synthetic_changepoint_scores(0, 10),
        lambda: synthetic_changepoint_scores(10, 0),
        lambda: synthetic_changepoint_scores(10, 10, scale=0.0),
        lambda: synthetic_changepoint_scores(10, 10, shift=np.nan),
    ],
)
def test_fail_closed_factories_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: first_crossing([], 20.0),
        lambda: first_crossing([np.nan], 20.0),
        lambda: first_crossing([-1.0], 0.5),
        lambda: first_crossing([1.0], 0.0),
        lambda: first_crossing([1.0], 1e301),
        lambda: first_crossing_log([np.inf], 20.0),
        lambda: power_log_bets([], 0.5),
        lambda: power_log_bets([1.2], 0.5),
        lambda: power_log_bets([np.nan], 0.5),
        lambda: power_log_bets([0.5], 0.0),
        lambda: power_log_bets([0.5], 1.5),
        lambda: power_log_bets([0.5], 0.5, "bogus"),  # type: ignore[arg-type]
        lambda: ctm_log_wealth([1.5], 0.5),
        lambda: cmm_log_wealth([0.5], "not-a-betting-class"),
        lambda: vovk_cusum_stat([]),
        lambda: vovk_cusum_stat([np.nan]),
        lambda: vovk_sr_stat([]),
        lambda: vovk_sr_stat([np.inf]),
        lambda: summarize_delays("m", "pfa", [None], changepoint=0, cap=5),
        lambda: summarize_delays("m", "pfa", [None], changepoint=10, cap=0),
        lambda: pfa_delay_study(0),
        lambda: pfa_delay_study(100, cap_multiple=0.5),
        lambda: pfa_delay_study(100, n_reps=0),
        lambda: pfa_delay_study(100, methods=("bogus",)),
        lambda: arl_delay_study(100, gamma=0.5),
        lambda: arl_delay_study(100, rho=1.5),
        lambda: arl_delay_study(100, methods=("bogus",)),
        lambda: delay_growth_study(()),
        lambda: delay_growth_study((0, 100)),
        lambda: false_alarm_study(regime="bogus"),  # type: ignore[arg-type]
        lambda: false_alarm_study(null="bogus"),  # type: ignore[arg-type]
        lambda: false_alarm_study(n_reps=0),
        lambda: false_alarm_study(horizon=0),
        lambda: false_alarm_study(horizon=20_001),
        lambda: arl_calibration_study(rho=0.0),
        lambda: arl_calibration_study(gamma=0.5),
        lambda: arl_calibration_study(n_reps=0),
        lambda: arl_calibration_study(early_ms=(0,)),
        lambda: arl_calibration_study(early_ms=(1000,)),  # >= horizon ceil(100/0.1)
    ],
)
def test_fail_closed_helpers_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()


def test_fail_closed_engine_raises() -> None:
    det = conformal_e_process(alpha=ALPHA, n_max=2, stat="sum")
    with pytest.raises(ValueError):
        det.update(np.nan)
    with pytest.raises(ValueError):
        det.update(np.inf)
    det.update(1.0)
    det.update(2.0)
    with pytest.raises(ValueError):  # past the declared horizon
        det.update(3.0)
    with pytest.raises(ValueError):
        det.update_p_value(1.5)
    with pytest.raises(ValueError):
        det.update_p_value(np.nan)
    det2 = conformal_e_process(alpha=ALPHA, n_max=2, stat="sum")
    with pytest.raises(ValueError):
        det2.run([1.0, 2.0, 3.0])  # stream longer than n_max
    with pytest.raises(ValueError):
        det2.run([])
    with pytest.raises(ValueError):
        det2.run([np.nan, 1.0])
    det3 = conformal_e_process(alpha=ALPHA, n_max=2, stat="sum")
    with pytest.raises(ValueError):
        det3.run_p_values([0.5, -0.1])
    with pytest.raises(ValueError):
        det3.run_p_values([0.5, 0.5, 0.5])
    bet = power_betting_class(thetas=(0.5,), sides="left")
    with pytest.raises(ValueError):
        bet.log_bets(1.5)
    with pytest.raises(ValueError):
        bet.log_bets(np.array([[0.5]]))
    with pytest.raises(ValueError):
        bet.log_bets([])
    # reset restores the initial state and the seed (run-after-reset identical)
    det4 = conformal_e_process(alpha=ALPHA, n_max=10, stat="sum", seed=3)
    r1 = det4.run(np.arange(1.0, 9.0), stop_on_alarm=False)
    r2 = det4.run(np.arange(1.0, 9.0), stop_on_alarm=False)
    np.testing.assert_array_equal(r1.statistic, r2.statistic)
    assert det4.value == float(r2.statistic[-1]) and det4.n_seen == 8
