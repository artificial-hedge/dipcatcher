"""Tests for quant_fund.microstructure.event_time_flow.

All numerics are SYNTHETIC correctness evidence: seeded LMF covering-model
sign streams and stochastic event clocks checked against exact closed forms
(the Hurwitz-zeta LMF autocorrelation, Poisson subordination mixtures) and
the paper's predicted exponent rescalings (Angstmann & Gebbie 2026,
arXiv:2609.13715). No market data, no PnL claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure import event_time_flow as etf

# ---------------------------------------------------------------------------
# Fixtures (seeded, module-scoped so Monte-Carlo cost is paid once)
# ---------------------------------------------------------------------------

N_EVENTS = 60_000
ALPHA = 1.5
BETA = 0.5
MU = 0.7


@pytest.fixture(scope="module")
def lmf() -> etf.LMFSigns:
    return etf.simulate_lmf_signs(N_EVENTS, ALPHA, 0.5, 11)


@pytest.fixture(scope="module")
def lags() -> np.ndarray:
    return np.unique(np.geomspace(4, 256, 10).astype(np.intp))


@pytest.fixture(scope="module")
def poisson_clock() -> etf.EventClock:
    return etf.EventClock(kind="poisson", rate=2.0)


@pytest.fixture(scope="module")
def pareto_clock() -> etf.EventClock:
    return etf.EventClock(kind="pareto", tail_exponent=MU, tail_scale=0.01)


@pytest.fixture(scope="module")
def pareto_paths(pareto_clock: etf.EventClock) -> list[np.ndarray]:
    return [pareto_clock.times(N_EVENTS, 900 + r) for r in range(12)]


@pytest.fixture(scope="module")
def pareto_anchored(lmf: etf.LMFSigns, pareto_paths: list[np.ndarray]) -> etf.EventAnchoredACF:
    """12-path pooled event-anchored calendar ACF under the fractional clock."""
    taus = np.geomspace(0.5, 500.0, 14)
    acc = np.zeros(taus.size)
    for tt in pareto_paths:
        acc += etf.event_anchored_sign_acf(lmf.signs, tt, taus).acf
    return etf.EventAnchoredACF(
        taus=taus,
        acf=acc / len(pareto_paths),
        mean_event_lags=np.full(taus.size, np.nan),
        n_pairs=0,
    )


# ---------------------------------------------------------------------------
# EventClock simulation and validation
# ---------------------------------------------------------------------------


def test_poisson_clock_times_and_gaps() -> None:
    clk = etf.EventClock(kind="poisson", rate=3.0)
    tt = clk.times(4000, 0)
    assert tt.size == 4000
    assert np.all(np.diff(tt) > 0.0)
    gaps = np.diff(tt)
    # exponential gaps: mean ~ 1/rate, CV ~ 1
    assert abs(float(gaps.mean()) - 1.0 / 3.0) < 0.05
    assert abs(float(gaps.std() / gaps.mean()) - 1.0) < 0.1


def test_clock_times_until_respects_horizon() -> None:
    clk = etf.EventClock(kind="poisson", rate=2.0)
    tt = clk.times_until(500.0, 1)
    assert np.all(tt <= 500.0)
    assert abs(tt.size - 1000.0) < 300.0


def test_clock_determinism() -> None:
    clk = etf.EventClock(kind="hawkes", hawkes_alpha=0.5)
    a = clk.times(500, 7)
    b = clk.times(500, 7)
    c = clk.times(500, 8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_regime_clock_clusters_and_rates() -> None:
    clk = etf.EventClock(kind="regime", regime_rates=(0.25, 4.0), stay_probs=(0.97, 0.97))
    tt = clk.times(6000, 3)
    # stationary mix rate = 1 / mean gap = 1 / sum pi_i / rate_i
    expected = 1.0 / (0.5 / 0.25 + 0.5 / 4.0)
    observed = tt.size / tt[-1]
    assert abs(observed - expected) / expected < 0.2
    # clustering: interarrival CV exceeds the exponential value 1
    gaps = np.diff(tt)
    assert float(gaps.std() / gaps.mean()) > 1.3


def test_hawkes_clock_overdispersed_and_rate() -> None:
    clk = etf.EventClock(kind="hawkes", hawkes_mu=0.4, hawkes_alpha=0.7)
    tt = clk.times(3000, 5)
    observed = tt.size / tt[-1]
    assert abs(observed - etf.hawkes_mean_intensity(0.4, 0.7)) < 0.4
    cm = etf.clock_count_moments(clk, np.array([100.0, 400.0]), n_paths=48, rng=2)
    assert np.all(cm.fano > 1.2)  # self-excitation -> overdispersion


def test_pareto_clock_count_scaling(pareto_clock: etf.EventClock) -> None:
    taus = np.geomspace(5.0, 2000.0, 10)
    cm = etf.clock_count_moments(pareto_clock, taus, n_paths=96, rng=4)
    fit = etf.fit_loglog_exponent(taus, cm.n_mean)
    assert 0.55 < fit.slope < 0.85  # E[N(tau)] ~ tau^0.7


def test_clock_fail_closed() -> None:
    with pytest.raises(ValueError):
        etf.EventClock(kind="nope")
    with pytest.raises(ValueError):
        etf.EventClock(kind="poisson", rate=0.0)
    with pytest.raises(ValueError):
        etf.EventClock(kind="regime", regime_rates=(1.0, 1.0))
    with pytest.raises(ValueError):
        etf.EventClock(kind="pareto", tail_exponent=1.0)
    with pytest.raises(ValueError):
        etf.EventClock(kind="hawkes", hawkes_alpha=1.0)
    with pytest.raises(ValueError):
        etf.EventClock(kind="hawkes", hawkes_alpha=-0.1)
    with pytest.raises(ValueError):
        etf.EventClock(kind="poisson", rate=1.0).times(0, 0)
    with pytest.raises(ValueError):
        etf.EventClock(kind="poisson", rate=1.0).times(8, None)  # unseeded


def test_mean_intensity_per_kind() -> None:
    assert etf.EventClock(kind="poisson", rate=2.0).mean_intensity() == 2.0
    hawk = etf.EventClock(kind="hawkes", hawkes_mu=0.4, hawkes_alpha=0.6)
    assert abs(hawk.mean_intensity() - 1.0) < 1e-12
    reg = etf.EventClock(kind="regime", regime_rates=(1.0, 3.0))
    assert abs(reg.mean_intensity() - 1.0 / (0.5 / 1.0 + 0.5 / 3.0)) < 1e-12


def test_hawkes_mean_intensity_closed_form() -> None:
    assert abs(etf.hawkes_mean_intensity(0.5, 0.5) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        etf.hawkes_mean_intensity(0.5, 1.0)


# ---------------------------------------------------------------------------
# LMF covering model: signs and the exact zeta autocorrelation
# ---------------------------------------------------------------------------


def test_lmf_signs_structure(lmf: etf.LMFSigns) -> None:
    assert lmf.signs.size == N_EVENTS
    assert set(np.unique(lmf.signs)) == {-1.0, 1.0}
    assert np.all(np.diff(lmf.meta_ids) >= 0)  # contiguous blocks
    assert lmf.n_metaorders > 1


def test_lmf_acf_theory_exact() -> None:
    ks = np.arange(0, 9)
    c = etf.lmf_acf_theory(ks, 1.5)
    assert c[0] == 1.0
    assert np.all(np.diff(c) < 0.0)
    # Hurwitz-zeta closed form: C(k) = zeta(alpha, k+1) / zeta(alpha)
    from scipy.special import zeta

    np.testing.assert_allclose(c, zeta(1.5, ks + 1.0) / zeta(1.5, 1.0))
    # asymptotic constant C(k) k^{0.5} -> 1 / ((alpha-1) zeta(alpha))
    kk = np.geomspace(1e4, 1e6, 6).astype(np.intp)
    asymp = etf.lmf_acf_theory(kk, 1.5) * kk**0.5
    assert np.all(np.abs(asymp - 1.0 / (0.5 * zeta(1.5, 1.0))) < 2e-3)


def test_lmf_sim_matches_theory(lmf: etf.LMFSigns, lags: np.ndarray) -> None:
    acf = etf.sign_autocorrelation(lmf.signs, lags)
    th = etf.lmf_acf_theory(lags, ALPHA)
    rel_l2 = float(np.linalg.norm(acf - th) / np.linalg.norm(th))
    assert rel_l2 < 0.25  # single-path scatter, honest tolerance


def test_lmf_sign_exponent_event_time(lmf: etf.LMFSigns, lags: np.ndarray) -> None:
    acf = etf.sign_autocorrelation(lmf.signs, lags)
    pos = acf > 0.0
    slope = etf.fit_loglog_exponent(lags[pos].astype(float), acf[pos]).slope
    # gamma = alpha - 1 = 0.5; single-path slope carries finite-n noise
    assert -0.85 < slope < -0.2


def test_lmf_fail_closed() -> None:
    with pytest.raises(ValueError):
        etf.simulate_lmf_signs(100, 1.0, 0.5, 0)
    with pytest.raises(ValueError):
        etf.simulate_lmf_signs(100, 2.0, 0.5, 0)
    with pytest.raises(ValueError):
        etf.simulate_lmf_signs(4, 1.5, 0.5, 0)
    with pytest.raises(ValueError):
        etf.simulate_lmf_signs(100, 1.5, 1.5, 0)
    with pytest.raises(ValueError):
        etf.simulate_lmf_signs(100, 1.5, 0.5, None)


def test_lmf_determinism() -> None:
    a = etf.simulate_lmf_signs(2000, 1.5, 0.5, 3)
    b = etf.simulate_lmf_signs(2000, 1.5, 0.5, 3)
    np.testing.assert_array_equal(a.signs, b.signs)
    np.testing.assert_array_equal(a.meta_ids, b.meta_ids)


def test_lmf_acf_theory_fail_closed() -> None:
    with pytest.raises(ValueError):
        etf.lmf_acf_theory(np.array([1, 2]), 0.5)
    with pytest.raises(ValueError):
        etf.lmf_acf_theory(np.array([1, 2]), 2.5)
    with pytest.raises(ValueError):
        etf.lmf_acf_theory(np.array([], dtype=np.intp), 1.5)
    with pytest.raises(ValueError):
        etf.lmf_acf_theory(np.array([-1]), 1.5)


def test_sign_autocorrelation_iid_and_fail_closed() -> None:
    rng = np.random.default_rng(0)
    white = rng.choice([-1.0, 1.0], size=20_000)
    acf = etf.sign_autocorrelation(white, np.array([1, 4, 16]))
    assert np.all(np.abs(acf) < 0.03)  # no memory for iid signs
    with pytest.raises(ValueError):
        etf.sign_autocorrelation(np.ones(100), np.array([1]))
    with pytest.raises(ValueError):
        etf.sign_autocorrelation(np.ones(100), np.array([0]))
    with pytest.raises(ValueError):
        etf.sign_autocorrelation(np.array([1.0, 2.0] * 50), np.array([1]))


# ---------------------------------------------------------------------------
# Operational-time propagator and kernel estimation
# ---------------------------------------------------------------------------


def test_propagator_kernel_values() -> None:
    g = etf.propagator_kernel(8, 2.0, 0.5)
    np.testing.assert_allclose(g, 2.0 / np.sqrt(np.arange(1, 9)))
    with pytest.raises(ValueError):
        etf.propagator_kernel(1)
    with pytest.raises(ValueError):
        etf.propagator_kernel(8, -1.0)
    with pytest.raises(ValueError):
        etf.propagator_kernel(8, 1.0, -0.1)


def test_metaorder_impact_curve_is_cumsum() -> None:
    g = etf.propagator_kernel(64, 1.0, BETA)
    imp = etf.metaorder_impact_curve(g)
    np.testing.assert_allclose(imp, np.cumsum(g))
    # square-root operational law: I(n) ~ n^{1-beta} (window past the
    # cumsum transient; local slope settles at 1 - beta)
    g_long = etf.propagator_kernel(1024, 1.0, BETA)
    imp_long = etf.metaorder_impact_curve(g_long)
    slope = etf.fit_loglog_exponent(np.arange(64, 1025, dtype=float), imp_long[63:]).slope
    assert abs(slope - 0.5) < 0.1
    with pytest.raises(ValueError):
        etf.metaorder_impact_curve(g, n_exec=100)


def test_propagator_path_noiseless_is_convolution() -> None:
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 1.0], size=512)
    g = etf.propagator_kernel(16, 1.0, BETA)
    path = etf.simulate_propagator_path(signs, g, 0.0, 0)
    # p_n = sum_{i <= n} eps_i G(n - i) — our convention (p_0 = 0)
    expected = np.concatenate(([0.0], np.convolve(signs, g)[:511]))
    np.testing.assert_allclose(path.prices, expected, atol=1e-12)
    np.testing.assert_allclose(path.moves, np.diff(path.prices), atol=1e-12)


def test_estimate_event_kernel_noiseless_recovery() -> None:
    rng = np.random.default_rng(2)
    signs = rng.choice([-1.0, 1.0], size=8192)
    g = etf.propagator_kernel(64, 1.0, BETA)
    path = etf.simulate_propagator_path(signs, g, 0.0, 0)
    g_hat = etf.estimate_event_kernel(path.prices, signs, 64)
    assert etf.kernel_rel_l2(g, g_hat) < 1e-10


def test_estimate_event_kernel_noisy_recovery() -> None:
    rng = np.random.default_rng(3)
    signs = rng.choice([-1.0, 1.0], size=16384)
    g = etf.propagator_kernel(32, 1.0, BETA)
    path = etf.simulate_propagator_path(signs, g, 0.02, 4)
    g_hat = etf.estimate_event_kernel(path.prices, signs, 32)
    assert etf.kernel_rel_l2(g, g_hat) < 0.3


def test_estimate_event_kernel_fail_closed() -> None:
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=64)
    prices = np.cumsum(signs) * 0.1
    with pytest.raises(ValueError):  # too few informative rows
        etf.estimate_event_kernel(prices, signs, 64)
    with pytest.raises(ValueError):  # size mismatch
        etf.estimate_event_kernel(prices[:-1], signs, 8)
    with pytest.raises(ValueError):  # prices not finite
        bad = prices.copy()
        bad[3] = np.nan
        etf.estimate_event_kernel(bad, signs, 8)


def test_kernel_rel_l2() -> None:
    g = etf.propagator_kernel(16)
    assert etf.kernel_rel_l2(g, g) == 0.0
    assert etf.kernel_rel_l2(g, 2.0 * g) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        etf.kernel_rel_l2(g, g[:8])


# ---------------------------------------------------------------------------
# Counting process and calendar subordination
# ---------------------------------------------------------------------------


def test_counting_process_steps() -> None:
    tt = np.array([0.5, 1.0, 2.5, 4.0])
    grid = np.array([0.0, 0.5, 1.5, 4.0, 9.0])
    np.testing.assert_array_equal(etf.counting_process(tt, grid), [0, 1, 2, 4, 4])
    with pytest.raises(ValueError):
        etf.counting_process(np.array([1.0]), grid)  # degenerate stream
    with pytest.raises(ValueError):
        etf.counting_process(tt, np.array([3.0, 1.0]))  # non-monotone grid


def test_subordinated_signs_tick_rule() -> None:
    tt = np.array([0.5, 1.0, 2.5, 4.0])
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    grid = np.array([0.5, 1.5, 3.0, 9.0])
    np.testing.assert_array_equal(etf.subordinated_signs(signs, tt, grid), [1.0, -1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        etf.subordinated_signs(signs, tt, np.array([0.1]))  # before first event


# ---------------------------------------------------------------------------
# Calendar-time sign ACF: distortions and closed-form checks
# ---------------------------------------------------------------------------


def test_calendar_acf_poisson_matches_mixture(lmf: etf.LMFSigns) -> None:
    """Uniform-probe ACF under Poisson equals the Pois(lambda*tau) mixture."""
    taus = np.geomspace(2.0, 500.0, 10)
    acc = np.zeros(taus.size)
    for r in range(6):
        tt = etf.EventClock(kind="poisson", rate=2.0).times(N_EVENTS, 60 + r)
        acc += etf.calendar_sign_acf(lmf.signs, tt, taus).acf
    acc /= 6
    cev = etf.lmf_acf_theory(np.arange(1, 4097), ALPHA)
    exact = etf.poisson_apparent_kernel(cev, taus, 2.0, zero_value=1.0)
    assert np.max(np.abs(acc - exact)) < 0.08
    rel_l2 = float(np.linalg.norm(acc - exact) / np.linalg.norm(exact))
    assert rel_l2 < 0.35


def test_calendar_acf_operational_remap_recovers_exponent(
    lmf: etf.LMFSigns, poisson_clock: etf.EventClock
) -> None:
    """Under a finite-mean clock the operational remap returns ~gamma."""
    tt = poisson_clock.times(N_EVENTS, 21)
    taus = np.geomspace(5.0, 1500.0, 10)
    caf = etf.calendar_sign_acf(lmf.signs, tt, taus)
    pos = caf.acf > 0.0
    cal_slope = etf.fit_loglog_exponent(caf.taus[pos], caf.acf[pos]).slope
    op_slope = etf.fit_loglog_exponent(caf.operational_lags[pos], caf.acf[pos]).slope
    # finite-mean clock: both windows see the same law; slope ~ -gamma = -0.5
    assert -0.95 < cal_slope < -0.25
    assert -0.95 < op_slope < -0.25
    assert abs(cal_slope - op_slope) < 0.3


def test_calendar_acf_pareto_plateau_distortion(
    lmf: etf.LMFSigns, pareto_clock: etf.EventClock
) -> None:
    """Uniform probes under the infinite-mean clock read a near-plateau."""
    acc = np.zeros(12)
    taus = np.geomspace(1.0, 2000.0, 12)
    for r in range(8):
        tt = pareto_clock.times(N_EVENTS, 70 + r)
        acc += etf.calendar_sign_acf(lmf.signs, tt, taus).acf
    acc /= 8
    pos = acc > 0.0
    slope = etf.fit_loglog_exponent(taus[pos], acc[pos]).slope
    # the apparent law is far shallower than the event-time -0.5
    assert -0.25 < slope < 0.05
    assert float(np.mean(acc[:4])) > 0.5


def test_calendar_acf_fail_closed(lmf: etf.LMFSigns) -> None:
    tt = etf.EventClock(kind="poisson", rate=1.0).times(500, 0)
    with pytest.raises(ValueError):  # horizon too short for taus
        etf.calendar_sign_acf(lmf.signs[:500], tt, np.array([400.0, 800.0]))
    with pytest.raises(ValueError):  # constant stream
        etf.calendar_sign_acf(np.ones(500), tt, np.array([1.0, 2.0, 4.0]))


# ---------------------------------------------------------------------------
# Event-anchored ACF and the operational-time correction
# ---------------------------------------------------------------------------


def test_event_anchored_equals_poisson_mixture(lmf: etf.LMFSigns) -> None:
    """eps_i vs eps_{N(t_i+tau)} pairs realize sum_k Pois(k) C_ev(k)."""
    taus = np.geomspace(1.0, 300.0, 10)
    acc = np.zeros(taus.size)
    for r in range(6):
        tt = etf.EventClock(kind="poisson", rate=2.0).times(N_EVENTS, 40 + r)
        acc += etf.event_anchored_sign_acf(lmf.signs, tt, taus).acf
    acc /= 6
    cev = etf.lmf_acf_theory(np.arange(1, 4097), ALPHA)
    exact = etf.poisson_apparent_kernel(cev, taus, 2.0, zero_value=1.0)
    assert np.max(np.abs(acc - exact)) < 0.1


def test_event_anchored_pareto_exponent_rescaling(
    pareto_anchored: etf.EventAnchoredACF,
) -> None:
    """Fractional clock rescales gamma -> mu*gamma = 0.7 * 0.5 = 0.35."""
    pos = pareto_anchored.acf > 0.0
    slope = etf.fit_loglog_exponent(pareto_anchored.taus[pos], pareto_anchored.acf[pos]).slope
    assert -0.55 < slope < -0.15


def test_event_anchored_vs_mixture_mc(pareto_anchored: etf.EventAnchoredACF) -> None:
    """Sim estimator tracks the MC origin-anchored mixture under pareto."""
    cev = etf.lmf_acf_theory(np.arange(1, 8193), ALPHA)
    par = etf.EventClock(kind="pareto", tail_exponent=MU, tail_scale=0.01)
    mc = etf.apparent_calendar_kernel(
        cev, par, pareto_anchored.taus, n_paths=96, rng=3, zero_value=1.0
    )
    assert np.max(np.abs(pareto_anchored.acf - mc.g_apparent)) < 0.2


def test_event_anchored_fail_closed(lmf: etf.LMFSigns) -> None:
    tt = np.array([1.0, 2.0])
    with pytest.raises(ValueError):  # signs/times size mismatch
        etf.event_anchored_sign_acf(lmf.signs[:10], tt, np.array([0.5]))
    short = etf.EventClock(kind="poisson", rate=1.0).times(8, 0)
    with pytest.raises(ValueError):  # fewer than 16 pairs
        etf.event_anchored_sign_acf(lmf.signs[:8], short, np.array([0.5, 1.0]))
    tt_f = np.array([0.0, 1e12])  # one giant gap, two events
    signs_f = np.array([1.0, -1.0])
    with pytest.raises(ValueError):  # too few pairs / frozen window
        etf.event_anchored_sign_acf(signs_f, tt_f, np.array([0.5]))


def test_operational_acf_recovers_event_law_poisson(lmf: etf.LMFSigns) -> None:
    """Binning calendar pairs by realized event lag returns C_ev(k)."""
    tt = etf.EventClock(kind="poisson", rate=2.0).times(N_EVENTS, 31)
    taus = np.geomspace(0.5, 1000.0, 12)
    op = etf.operational_sign_acf(lmf.signs, tt, taus, k_max=200)
    th = etf.lmf_acf_theory(np.arange(201), ALPHA)
    sel = (op.pair_counts > 1000) & (op.event_lags >= 4) & (op.event_lags <= 64)
    sel &= ~np.isnan(op.acf)
    est, ref = op.acf[sel], th[op.event_lags[sel]]
    rel_l2 = float(np.linalg.norm(est - ref) / np.linalg.norm(ref))
    assert rel_l2 < 0.3  # finite-stream truncation gives a mild low bias
    slope = etf.fit_loglog_exponent(
        op.event_lags[sel].astype(float), np.clip(est, 1e-6, None)
    ).slope
    # truncation steepens the recovered law; assert a decaying power fit
    assert -1.1 < slope < -0.25


def test_operational_acf_pareto_bulk_recovery(
    lmf: etf.LMFSigns, pareto_paths: list[np.ndarray]
) -> None:
    """Under the fractional clock the conditioned law still decays ~k^-0.5
    on the well-populated bins (tail bins thin out — documented limit)."""
    km = 256
    taus = np.geomspace(0.5, 2000.0, 16)
    sumw = np.zeros(km + 1)
    cnt = np.zeros(km + 1)
    for tt in pareto_paths:
        op = etf.operational_sign_acf(lmf.signs, tt, taus, k_max=km)
        w = np.where(np.isnan(op.acf), 0.0, op.acf * op.pair_counts)
        sumw += w
        cnt += op.pair_counts
    pooled = np.where(cnt > 0, sumw / np.maximum(cnt, 1), np.nan)
    sel = (cnt > 500) & (np.arange(km + 1) >= 2) & (np.arange(km + 1) <= 64)
    est = pooled[sel]
    # giant-gap clustering leaves heavy per-bin noise even at 12 pooled
    # paths — assert bulk statistics, not bin-level exactness
    assert np.mean(est > 0.0) > 0.75
    th = etf.lmf_acf_theory(np.arange(km + 1)[sel], ALPHA)
    assert 0.3 < float(np.mean(est) / np.mean(th)) < 2.5
    assert float(np.median(est)) > 0.0
    # and it still clearly decays with event lag (first vs second half means)
    assert float(np.mean(est[: len(est) // 2])) > float(np.mean(est[len(est) // 2 :]))


def test_operational_acf_fail_closed(lmf: etf.LMFSigns) -> None:
    tt = etf.EventClock(kind="poisson", rate=1.0).times(512, 0)
    with pytest.raises(ValueError):  # k_max >= n
        etf.operational_sign_acf(lmf.signs[:512], tt, np.array([1.0, 4.0]), 512)
    with pytest.raises(ValueError):  # horizon too short
        etf.operational_sign_acf(lmf.signs[:512], tt, np.array([400.0]), 8)
    frozen = np.array([10.0, 20.0])  # two events, giant gaps
    signs2 = np.array([1.0, -1.0])
    with pytest.raises(ValueError):  # too few events to probe
        etf.operational_sign_acf(signs2, frozen, np.array([0.5]), 1)


# ---------------------------------------------------------------------------
# Apparent (subordinated) kernels: MC vs closed form, exponent rescaling
# ---------------------------------------------------------------------------


def test_apparent_kernel_poisson_mc_vs_exact() -> None:
    g = etf.propagator_kernel(64, 1.0, BETA)
    taus = np.geomspace(0.5, 60.0, 10)
    pois = etf.EventClock(kind="poisson", rate=2.0)
    mc = etf.apparent_calendar_kernel(g, pois, taus, n_paths=256, rng=5)
    exact = etf.poisson_apparent_kernel(g, taus, 2.0)
    rel = np.abs(mc.g_apparent - exact) / np.maximum(np.abs(exact), 1e-12)
    assert float(rel.max()) < 0.15
    np.testing.assert_allclose(mc.n_mean, 2.0 * taus, rtol=0.1)


def test_apparent_kernel_poisson_preserves_exponent() -> None:
    g = etf.propagator_kernel(1024, 1.0, BETA)
    taus = np.geomspace(20.0, 400.0, 8)
    exact = etf.poisson_apparent_kernel(g, taus, 2.0)
    slope = etf.fit_loglog_exponent(taus, exact).slope
    # finite-mean clock: G_cal(tau) ~ tau^{-beta} = tau^{-0.5}
    assert -0.65 < slope < -0.35


def test_apparent_kernel_pareto_rescales_exponent() -> None:
    g = etf.propagator_kernel(4096, 1.0, BETA)
    taus = np.geomspace(0.5, 500.0, 12)
    par = etf.EventClock(kind="pareto", tail_exponent=MU, tail_scale=0.01)
    mc = etf.apparent_calendar_kernel(g, par, taus, n_paths=128, rng=6)
    pos = mc.g_apparent > 0.0
    slope = etf.fit_loglog_exponent(taus[pos], mc.g_apparent[pos]).slope
    # mu * beta = 0.35
    assert -0.5 < slope < -0.2


def test_apparent_kernel_fail_closed() -> None:
    g = etf.propagator_kernel(16)
    with pytest.raises(ValueError):
        etf.apparent_calendar_kernel(
            g, etf.EventClock(kind="poisson"), np.array([1.0, 2.0]), n_paths=3
        )
    with pytest.raises(ValueError):
        etf.poisson_apparent_kernel(g, np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        etf.poisson_apparent_kernel(-g, np.array([1.0]), 1.0)


def test_poisson_apparent_kernel_exact_limits() -> None:
    g = etf.propagator_kernel(8, 1.0, BETA)
    small = etf.poisson_apparent_kernel(g, np.array([1e-9]), 1.0)
    assert abs(float(small[0])) < 1e-3  # N ~ 0 -> g(0) = 0
    big = etf.poisson_apparent_kernel(g, np.array([1e9]), 1.0)
    assert abs(float(big[0]) - g[-1]) < 1e-6  # N >> L -> clamps to g_L


# ---------------------------------------------------------------------------
# Calendar impact profile
# ---------------------------------------------------------------------------


def test_calendar_impact_poisson_sqrt_law() -> None:
    g = etf.propagator_kernel(1024, 1.0, BETA)
    taus = np.geomspace(5.0, 300.0, 8)
    prof = etf.calendar_impact_profile(
        g, etf.EventClock(kind="poisson", rate=2.0), 0.5, taus, n_paths=64, rng=7
    )
    ok = (prof.impact_mean > 0.0) & (prof.n_mean < g.size)
    slope = etf.fit_loglog_exponent(taus[ok], prof.impact_mean[ok]).slope
    assert 0.35 < slope < 0.7  # ~ sqrt (1 - beta)


def test_calendar_impact_pareto_rescaled_exponent(pareto_clock) -> None:  # noqa: ANN001
    g = etf.propagator_kernel(1024, 1.0, BETA)
    taus = np.geomspace(10.0, 3e4, 10)
    prof = etf.calendar_impact_profile(g, pareto_clock, 0.2, taus, n_paths=96, rng=8)
    ok = (prof.impact_mean > 0.0) & (prof.n_mean < g.size)
    slope = etf.fit_loglog_exponent(taus[ok], prof.impact_mean[ok]).slope
    # mu * (1 - beta) = 0.35
    assert 0.2 < slope < 0.55


def test_calendar_impact_fail_closed(pareto_clock) -> None:  # noqa: ANN001
    g = etf.propagator_kernel(32)
    with pytest.raises(ValueError):  # participation out of (0, 1]
        etf.calendar_impact_profile(g, pareto_clock, 0.0, np.array([1.0, 2.0]), n_paths=8, rng=0)
    with pytest.raises(ValueError):
        etf.calendar_impact_profile(g, pareto_clock, 1.5, np.array([1.0, 2.0]), n_paths=8, rng=0)


# ---------------------------------------------------------------------------
# Counting-process moments, imbalance, boundaries, stream utilities
# ---------------------------------------------------------------------------


def test_poisson_count_moments_exact() -> None:
    taus = np.array([5.0, 25.0])
    cm = etf.poisson_count_moments(taus, 3.0)
    np.testing.assert_allclose(cm.n_mean, 3.0 * taus)
    np.testing.assert_allclose(cm.n_var, 3.0 * taus)
    np.testing.assert_allclose(cm.fano, 1.0)


def test_clock_count_moments_poisson_agreement() -> None:
    taus = np.array([10.0, 60.0])
    mc = etf.clock_count_moments(etf.EventClock(kind="poisson", rate=2.5), taus, n_paths=64, rng=9)
    np.testing.assert_allclose(mc.n_mean, 2.5 * taus, rtol=0.15)
    assert np.all(np.abs(mc.fano - 1.0) < 0.4)


def test_clock_count_moments_fail_closed() -> None:
    with pytest.raises(ValueError):
        etf.clock_count_moments(etf.EventClock(kind="poisson"), np.array([1.0]), n_paths=3, rng=0)


def test_calendar_imbalance_conserves_counts() -> None:
    signs = np.ones(500)
    tt = etf.EventClock(kind="poisson", rate=2.0).times(500, 10)
    edges, imb = etf.calendar_imbalance(signs, tt, 20.0)
    assert edges.size == imb.size + 1
    assert abs(float(imb.sum()) - 500.0) < 1e-9
    with pytest.raises(ValueError):
        etf.calendar_imbalance(signs, tt, 0.0)


def test_activity_boundaries_find_regime_switch() -> None:
    """Two concatenated poisson streams at different rates -> boundary near
    the concatenation time."""
    slow = etf.EventClock(kind="poisson", rate=0.2).times_until(500.0, 12)
    fast = etf.EventClock(kind="poisson", rate=4.0).times_until(125.0, 13)
    tt = np.concatenate([slow, fast + 500.0])
    ab = etf.activity_rate_boundaries(tt, bucket_dt=10.0, min_size=4)
    assert ab.changepoint_times.size >= 1
    assert np.min(np.abs(ab.changepoint_times - 500.0)) < 60.0


def test_activity_boundaries_fail_closed() -> None:
    tt = np.array([0.5, 1.0, 1.5])  # too few events for buckets
    with pytest.raises(ValueError):
        etf.activity_rate_boundaries(tt, bucket_dt=1.0, min_size=32)


def test_interarrival_stats_and_fail_closed() -> None:
    tt = etf.EventClock(kind="poisson", rate=2.0).times(2000, 14)
    st = etf.interarrival_stats(tt)
    assert st["n_gaps"] == 1999.0
    assert abs(st["cv_gap"] - 1.0) < 0.15
    with pytest.raises(ValueError):
        etf.interarrival_stats(np.array([1.0]))  # no gaps
    with pytest.raises(ValueError):
        etf.interarrival_stats(np.arange(10.0))  # constant gaps


def test_zi_lob_event_stream_composition() -> None:
    es = etf.zi_lob_event_stream(300.0)
    assert es.times.size > 4
    assert np.all(np.diff(es.times) > 0.0)
    assert set(np.unique(es.signs)) <= {-1.0, 1.0}
    assert es.label == "zi_lob"


# ---------------------------------------------------------------------------
# Loglog fitter and bench
# ---------------------------------------------------------------------------


def test_fit_loglog_exponent_exact_power_law() -> None:
    x = np.geomspace(1.0, 1000.0, 20)
    y = 3.0 * x**-0.42
    fit = etf.fit_loglog_exponent(x, y)
    assert abs(fit.slope + 0.42) < 1e-8
    assert fit.r2 > 0.999
    assert fit.n == 20


def test_fit_loglog_exponent_fail_closed() -> None:
    with pytest.raises(ValueError):
        etf.fit_loglog_exponent(np.array([1.0, -2.0, 3.0]), np.ones(3))
    with pytest.raises(ValueError):
        etf.fit_loglog_exponent(np.ones(4), np.ones(4))  # zero log-range
    with pytest.raises(ValueError):
        etf.fit_loglog_exponent(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_bench_event_time_flow() -> None:
    out = etf.bench_event_time_flow(seed=0)
    assert all(k.startswith("synthetic_") for k in out)
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_kernel_l2_rel_error"] < 1e-10
    assert out["synthetic_sign_acf_l2_vs_theory"] < 0.6
    assert out["synthetic_apparent_kernel_mc_max_rel_err"] < 0.2
    assert out["synthetic_impact_slope_calendar_pareto"] < out["synthetic_impact_slope_operational"]
    assert (
        abs(
            out["synthetic_sign_slope_calendar_pareto"]
            - out["synthetic_sign_slope_calendar_theory"]
        )
        < 0.2
    )


def test_bench_determinism() -> None:
    a = etf.bench_event_time_flow(seed=0)
    b = etf.bench_event_time_flow(seed=0)
    assert a == b
