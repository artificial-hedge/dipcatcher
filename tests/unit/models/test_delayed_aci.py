"""Tests for quant_fund.models.delayed_aci — τ-DACI (wave-13 theory lane A8).

Reference: El Halabi & Brandt, 2026, "Adaptive Conformal Inference Under
Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic",
arXiv:2609.07251. All streams are seeded SYNTHETIC data — correctness tests,
never market evidence (AGENTS.md honesty contract). Only coverage and
interval-score (proper score) diagnostics are asserted.

Brief-vs-paper correction (verified against the fetched paper): the strong
"r organizes performance" claim is about the INTERVAL SCORE curve collapse
(paper Sec. 7.2 / Table 6: ≈79% within-bin scatter reduction under AR(1));
the paper explicitly notes that worst-case LOCAL coverage-error curves do
NOT collapse across persistence levels under r. The AR(1)-drift tests below
therefore assert (a) per-persistence monotone ordering of the coverage
deviation in r (τ swept at fixed L, so r ∝ τ) and (b) the pooled
curve-collapse of the interval score under r versus raw τ.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from quant_fund.models.conformal import AdaptiveConformal
from quant_fund.models.delayed_aci import (
    DelayedACI,
    adaptation_duration,
    curve_collapse_reduction,
    delay_to_memory_ratio,
    estimate_delay_to_memory,
    long_run_coverage_bound,
    marginal_coverage_bound,
    memory_length_ar1,
    memory_length_garch,
    memory_length_markov,
    optimal_gamma,
    run_delayed_conformal,
    worst_case_local_coverage_error,
)

ALPHA = 0.1
NOMINAL = 1.0 - ALPHA


# ------------------------------------------------------------------ streams
def _ar1(n: int, phi: float, rng: np.random.Generator, sd: float = 1.0) -> np.ndarray:
    """Stationary AR(1) with marginal sd ``sd`` (innovation sd sd*sqrt(1-phi^2))."""
    sigma_eta = sd * np.sqrt(1.0 - phi * phi)
    x = np.empty(n, dtype=float)
    x[0] = rng.normal(0.0, sd)
    eta = rng.normal(0.0, sigma_eta, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eta[t]
    return x


def _garch(n: int, arch_alpha: float, arch_beta: float, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) residuals, omega = 1 - alpha - beta (unit unconditional var)."""
    omega = 1.0 - arch_alpha - arch_beta
    s2 = np.empty(n, dtype=float)
    e2 = np.empty(n, dtype=float)
    s2[0] = 1.0
    z = rng.normal(0.0, 1.0, n)
    for t in range(n):
        if t > 0:
            s2[t] = omega + arch_alpha * e2[t - 1] + arch_beta * s2[t - 1]
        e2[t] = s2[t] * z[t] ** 2
    return np.sign(z) * np.sqrt(e2)


def _drift_stream(n: int, phi: float, seed: int, sigma_d: float = 1.0, sigma_z: float = 0.5):
    """AR(1)-drift residual stream: eps_t = d_t + noise, d_t AR(1) with memory L."""
    rng = np.random.default_rng(seed)
    drift = _ar1(n, phi, rng, sd=sigma_d)
    return drift + rng.normal(0.0, sigma_z, n)


def _rolling_cov(covered: np.ndarray, k: int = 100) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(covered)])
    return (cs[k:] - cs[:-k]) / k


def _rms_coverage_deviation(covered: np.ndarray, k: int = 100, skip: int = 200) -> float:
    """RMS deviation of rolling-k empirical coverage from the nominal 1 - alpha."""
    rc = _rolling_cov(covered[skip:], k)
    return float(np.sqrt(np.mean((rc - NOMINAL) ** 2)))


# ---------------------------------------------------- AR(1)-drift MC fixture
SWEEP_T = 3000
SWEEP_WINDOW = 200
SWEEP_GAMMA = 0.02
SWEEP_PHIS = (0.7, 0.9, 0.98)
SWEEP_TAUS = (1, 6, 18)
SWEEP_SEEDS = (0, 1, 2)


def _sweep_run(phi: float, tau: int, seed: int) -> tuple[float, float]:
    eps = _drift_stream(SWEEP_T + tau, phi, 1000 * seed + 17)
    res = run_delayed_conformal(eps, tau, alpha=ALPHA, gamma=SWEEP_GAMMA, window=SWEEP_WINDOW)
    dev = _rms_coverage_deviation(res.covered)
    return dev, res.mean_interval_score


@pytest.fixture(scope="module")
def ar1_drift_sweep() -> tuple[list[dict[str, float]], float]:
    """Seeded (phi, tau) sweep + iid Gaussian baseline with matched marginal sd.

    Baseline: exchangeable N(0, sqrt(sigma_d^2 + sigma_z^2)) residuals — the
    AR(1)-drift streams all share this marginal sd, so interval scores are
    directly comparable (the paper's unit-variance normalization, Sec. 6.3).
    """
    rows: list[dict[str, float]] = []
    for phi in SWEEP_PHIS:
        length = memory_length_ar1(phi)
        for tau in SWEEP_TAUS:
            devs, iscs = [], []
            for seed in SWEEP_SEEDS:
                dev, isc = _sweep_run(phi, tau, seed)
                devs.append(dev)
                iscs.append(isc)
            rows.append(
                {
                    "phi": phi,
                    "tau": float(tau),
                    "L": length,
                    "r": delay_to_memory_ratio(tau, length),
                    "dev": float(np.mean(devs)),
                    "isc": float(np.mean(iscs)),
                }
            )
    baseline = []
    sd_stream = np.sqrt(1.0 + 0.5**2)
    for seed in SWEEP_SEEDS:
        rng = np.random.default_rng(90000 + seed)
        eps = rng.normal(0.0, sd_stream, SWEEP_T + 40)
        res = run_delayed_conformal(eps, 5, alpha=ALPHA, gamma=SWEEP_GAMMA, window=SWEEP_WINDOW)
        baseline.append(res.mean_interval_score)
    return rows, float(np.mean(baseline))


# ------------------------------------------------- tau = 1 exactness (pin)
def test_tau1_matches_adaptive_conformal_exactly() -> None:
    """τ = 1 must reproduce the repo's AdaptiveConformal recursion bit-for-bit.

    Same closed-form update alpha_{t+1} = clip(alpha_t + gamma*(alpha - err_t),
    1e-3, 1 - 1e-3) (Gibbs & Candès, 2021; paper Eq. (6) at τ = 1), same clip
    box, same float ops — exact equality, not tolerance. The err stream is
    seeded with mean > alpha so the level walks into the clip box and the pin
    exercises the clipped edges too.
    """
    rng = np.random.default_rng(0)
    errs = (rng.random(500) < 0.12).astype(float)
    aci = AdaptiveConformal(alpha=ALPHA, gamma=0.05)
    ref = [aci.alpha_t]
    for e in errs:
        aci.update(float(e))
        ref.append(aci.alpha_t)
    daci = DelayedACI(alpha=ALPHA, gamma=0.05, tau=1)
    issued = [daci.step(None)]
    for e in errs:
        issued.append(daci.step(float(e)))
    assert issued == ref  # exact float equality of the full level path
    assert np.array_equal(daci.alpha_history, np.asarray(ref, dtype=float))
    assert daci.clip_updates_ > 0  # the clipped edges were exercised
    assert np.array_equal(daci.feedback_history[1:], errs)
    assert np.isnan(daci.feedback_history[0])


def test_tau1_closed_form_identity_and_telescoping() -> None:
    """Unclipped regime: per-step closed form is exact and errors telescope.

    The error stream is the balanced pattern [0 x9, 1] tiled (miscoverage rate
    exactly alpha), so the level is a zero-drift walk that provably stays
    inside the clip box; the telescoping sum Σ(e_m − α) = (α_1 − α_{K+1})/γ
    is the exact identity behind the paper's long-run bound (App. A.2, Step 1).
    """
    errs = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float), 30)
    gamma = 0.01
    daci = DelayedACI(alpha=ALPHA, gamma=gamma, tau=1)
    daci.step(None)
    for e in errs:
        daci.step(float(e))
    assert daci.clip_updates_ == 0
    levels = daci.alpha_history
    ref = ALPHA
    assert levels[0] == ref
    for k, e in enumerate(errs):
        ref = float(np.clip(ref + gamma * (ALPHA - e), 1e-3, 1.0 - 1e-3))
        assert levels[k + 1] == ref  # closed-form recursion, exact
    consumed = daci.feedback_history[1:]
    assert np.allclose(np.diff(levels), gamma * (ALPHA - consumed), rtol=0.0, atol=1e-15)
    assert abs(float(np.sum(consumed - ALPHA)) - (levels[0] - levels[-1]) / gamma) < 1e-9


def test_phase_decomposition_is_interleaved_aci() -> None:
    """τ = 3: the issued path is exactly 3 interleaved one-step ACI sequences.

    Paper Eq. (7)-(8): phase θ collects steps s ≡ θ (mod τ); within a phase
    the delayed recursion is the ordinary ACI update α_{m+1} = α_m + γ(α − e_m)
    with no explicit delay.
    """
    tau, gamma, n_steps = 3, 0.01, 60
    rng = np.random.default_rng(11)
    errs = (rng.random(n_steps) < 0.12).astype(float)
    daci = DelayedACI(alpha=ALPHA, gamma=gamma, tau=tau)
    for s in range(1, n_steps + 1):
        daci.step(None if s <= tau else float(errs[s - 2]))
    assert daci.clip_updates_ == 0
    levels = daci.alpha_history
    phases = daci.phases_
    assert len(phases) == tau
    merged = np.empty(n_steps, dtype=float)
    for p, (lv, er) in enumerate(phases):
        assert lv.size == len(range(p, n_steps, tau))
        assert er.size == lv.size - 1
        merged[p::tau] = lv
        # exact phase-local ACI recursion, recomputed independently
        ref = float(lv[0])
        assert ref == ALPHA  # Algorithm 2: every phase starts at alpha
        for m, e in enumerate(er):
            ref = float(np.clip(ref + gamma * (ALPHA - e), 1e-3, 1.0 - 1e-3))
            assert lv[m + 1] == ref
        # phase errors are the global feedback subsequence for those steps
        assert np.array_equal(er, daci.feedback_history[p::tau][1:])
    assert np.array_equal(merged, levels)


# ----------------------------------------------------- fail-closed edges
def test_step_feedback_schedule_fail_closed() -> None:
    """Delayed feedback is guaranteed but scheduled: due exactly after τ steps."""
    daci = DelayedACI(alpha=ALPHA, gamma=0.05, tau=3)
    with pytest.raises(ValueError, match="None"):
        daci.step(0.0)  # nothing in flight yet
    daci.step(None)
    daci.step(None)
    with pytest.raises(ValueError, match="None"):
        daci.step(1.0)  # still only 2 in flight
    daci.step(None)  # queue now full (3 in flight)
    with pytest.raises(ValueError, match="required"):
        daci.step(None)  # feedback for the oldest set is due
    for bad in (1.5, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="err"):
            daci.step(bad)
    assert daci.step(1.0) == pytest.approx(
        float(np.clip(ALPHA + 0.05 * (ALPHA - 1.0), 1e-3, 1.0 - 1e-3))
    )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_bad_alpha_fail_closed(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        DelayedACI(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        run_delayed_conformal(np.zeros(50), 1, alpha=alpha, window=10)
    with pytest.raises(ValueError, match="alpha"):
        long_run_coverage_bound(alpha, 0.01, 1, 100)


@pytest.mark.parametrize("gamma", [0.0, -0.01, np.nan, np.inf])
def test_bad_gamma_fail_closed(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        DelayedACI(gamma=gamma)
    with pytest.raises(ValueError, match="gamma"):
        long_run_coverage_bound(ALPHA, gamma, 1, 100)
    with pytest.raises(ValueError, match="gamma"):
        marginal_coverage_bound(1.0, gamma, 0.01)


@pytest.mark.parametrize("tau", [0, -1, 1.5, np.int64(0)])
def test_bad_tau_fail_closed(tau: object) -> None:
    with pytest.raises(ValueError, match="tau"):
        DelayedACI(tau=tau)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau"):
        run_delayed_conformal(np.zeros(50), tau, window=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau"):
        delay_to_memory_ratio(tau, 5.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("clip", [(0.0, 0.5), (0.5, 0.5), (0.9, 0.1), (0.0, 1.0), (np.nan, 0.5)])
def test_bad_clip_fail_closed(clip: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="clip"):
        DelayedACI(clip=clip)
    with pytest.raises(ValueError, match="clip"):
        run_delayed_conformal(np.zeros(50), 1, window=10, clip=clip)


def test_bound_input_validation_fail_closed() -> None:
    for bad_t in (0, -5, 2.5):
        with pytest.raises(ValueError, match="T"):
            long_run_coverage_bound(ALPHA, 0.01, 2, bad_t)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha_init"):
        long_run_coverage_bound(ALPHA, 0.01, 3, 100, alpha_init=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="alpha_init"):
        long_run_coverage_bound(ALPHA, 0.01, 2, 100, alpha_init=np.array([-0.1, 0.2]))
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="lip"):
            marginal_coverage_bound(bad, 0.01, 0.01)
        with pytest.raises(ValueError, match="level_shift"):
            optimal_gamma(bad)
    with pytest.raises(ValueError, match="level_shift"):
        marginal_coverage_bound(1.0, 0.01, -0.5)


# ------------------------------------------------------- bound diagnostics
def test_long_run_bound_formula_and_tau1_reduction() -> None:
    """Eq. (11)/(33) with default init: tau*max(a,1-a)/(gamma*T) + tau/T."""
    gamma, T = 0.02, 2000.0
    for tau in (1, 5, 20):
        expected = tau * max(ALPHA, 1 - ALPHA) / (gamma * T) + tau / T
        assert long_run_coverage_bound(ALPHA, gamma, tau, int(T)) == pytest.approx(expected)
    # tau = 1 is exactly Gibbs & Candes (2021) Prop 4.1's bound form
    b1 = long_run_coverage_bound(ALPHA, gamma, 1, 1000)
    assert b1 == pytest.approx(max(ALPHA, 1 - ALPHA) / (gamma * 1000) + 1 / 1000)
    # explicit tau dependence: linear in tau, decreasing in T
    assert (
        long_run_coverage_bound(ALPHA, gamma, 1, 1000)
        < long_run_coverage_bound(ALPHA, gamma, 5, 1000)
        < long_run_coverage_bound(ALPHA, gamma, 20, 1000)
    )
    assert long_run_coverage_bound(ALPHA, gamma, 5, 4000) < long_run_coverage_bound(
        ALPHA, gamma, 5, 1000
    )
    # per-phase initialization enters through sum_theta max{a1, 1-a1}
    inits = np.array([0.1, 0.5, 0.9])
    b = long_run_coverage_bound(ALPHA, gamma, 3, 500, alpha_init=inits)
    assert b == pytest.approx((0.9 + 0.5 + 0.9) / (gamma * 500) + 3 / 500)
    assert long_run_coverage_bound(ALPHA, gamma, 3, 500, alpha_init=0.4) == pytest.approx(
        3 * 0.6 / (gamma * 500) + 3 / 500
    )


BOUND_CONFIGS = (("iid", 1, 0.02), ("iid", 5, 0.01), ("ar1", 10, 0.02), ("shift", 20, 0.01))


@pytest.mark.parametrize("kind,tau,gamma", BOUND_CONFIGS)
def test_long_run_bound_holds_on_synthetic_streams(kind: str, tau: int, gamma: float) -> None:
    """SYNTHETIC: realized |mean err − α| never exceeds the Eq. (11) bound.

    The bound is checked at every prefix T of each run. The clip box must stay
    inactive (asserted): with all updates interior, the phase-wise telescoping
    of App. A.2 is exact, so the bound is a theorem of the realized path, not
    a statistical hope.
    """
    rng = np.random.default_rng(31)
    if kind == "iid":
        eps = rng.normal(0.0, 1.0, 3000)
    elif kind == "ar1":
        eps = _ar1(3000, 0.9, rng)
    else:
        eps = np.concatenate([rng.normal(0.0, 1.0, 1500), rng.normal(0.0, 3.0, 1500)])
    res = run_delayed_conformal(eps, tau, alpha=ALPHA, gamma=gamma, window=250)
    assert res.clip_updates == 0
    assert np.all(res.levels > 1e-3) and np.all(res.levels < 1.0 - 1e-3)
    n_c = res.err.size
    for frac in (0.25, 0.5, 0.75, 1.0):
        T = int(frac * n_c)
        dev = abs(float(np.mean(res.err[:T])) - ALPHA)
        assert dev <= long_run_coverage_bound(ALPHA, gamma, tau, T) + 1e-12, (
            f"{kind} tau={tau} gamma={gamma} T={T}: dev={dev:.5f}"
        )
    assert res.long_run_deviation <= res.coverage_bound + 1e-12


def test_marginal_bound_formula_and_optimal_gamma() -> None:
    """Eq. (12)/(50) closed form; γ_opt = sqrt(2·shift) minimizes it (Sec. 4.4)."""
    lip, shift = 3.0, 0.01
    gamma = 0.05
    expected = lip * (1 + gamma) / gamma * shift + 0.5 * lip * gamma
    assert marginal_coverage_bound(lip, gamma, shift) == pytest.approx(expected)
    # zero environment change across the horizon leaves only the adaptation penalty
    assert marginal_coverage_bound(lip, gamma, 0.0) == pytest.approx(0.5 * lip * gamma)
    g_opt = optimal_gamma(shift)
    assert g_opt == pytest.approx(np.sqrt(2.0 * shift))
    best = marginal_coverage_bound(lip, g_opt, shift)
    for g in (0.001, 0.005, 0.02, 0.05, 0.2, 0.5, 1.0):
        assert marginal_coverage_bound(lip, g, shift) > best


# ------------------------------------------- delay-to-memory ratio helpers
def test_memory_lengths_match_paper_tables() -> None:
    """Closed forms pinned against the paper's stated values (Sec. 6.3-6.6)."""
    # AR(1): phi in {0.1, 0.6, 0.85, 0.95, 0.99} -> L ~= {0.43, 1.96, 6.15, 19.50, 99.50}
    for phi, expected in zip(
        (0.1, 0.6, 0.85, 0.95, 0.99), (0.43, 1.96, 6.15, 19.50, 99.50), strict=True
    ):
        assert memory_length_ar1(phi) == pytest.approx(expected, abs=0.005)
    # GARCH(1,1): a+b in {0.5, 0.9, 0.95, 0.99} -> L_vol ~= {1.44, 9.49, 19.50, 99.50}
    for ab, expected in zip((0.5, 0.9, 0.95, 0.99), (1.44, 9.49, 19.50, 99.50), strict=True):
        assert memory_length_garch(0.1, ab - 0.1) == pytest.approx(expected, abs=0.005)
    # Markov switching: symmetric p -> lambda_2 = 2p - 1,
    # L_regime ~= {1.96, 4.48, 9.49, 16.16, 49.50} (paper Sec. 6.6)
    for p, expected in zip(
        (0.80, 0.90, 0.95, 0.97, 0.99), (1.96, 4.48, 9.49, 16.16, 49.50), strict=True
    ):
        assert memory_length_markov(p, p) == pytest.approx(expected, abs=0.005)
    # ratio: r = tau / L, surviving dependence e^{-r}
    r = delay_to_memory_ratio(20, memory_length_ar1(0.95))
    assert r == pytest.approx(20.0 / 19.4957, abs=1e-3)
    assert np.exp(-r) == pytest.approx(0.95**20, abs=1e-3)


def test_memory_length_fail_closed_edges() -> None:
    for bad in (0.0, 1.0, -0.5, 1.5, np.nan):
        with pytest.raises(ValueError, match="phi"):
            memory_length_ar1(bad)
    with pytest.raises(ValueError, match="arch_alpha"):
        memory_length_garch(-0.1, 0.5)
    with pytest.raises(ValueError, match="arch_alpha \\+ arch_beta"):
        memory_length_garch(0.5, 0.6)  # non-stationary
    with pytest.raises(ValueError, match="lambda_2"):
        memory_length_markov(0.5, 0.5)  # lambda_2 = 0: no regime memory
    with pytest.raises(ValueError, match="p00 and p11"):
        memory_length_markov(1.0, 0.9)
    with pytest.raises(ValueError, match="memory_length"):
        delay_to_memory_ratio(5, 0.0)
    with pytest.raises(ValueError, match="memory_length"):
        delay_to_memory_ratio(5, -2.0)


def test_estimate_delay_to_memory_ar1_levels() -> None:
    """SYNTHETIC: log-ACF decay regression recovers L = -1/log phi and orders it."""
    for seed in (101, 102):
        lengths: dict[float, float] = {}
        for phi in (0.6, 0.85, 0.95):
            rng = np.random.default_rng(seed)
            x = _ar1(6000, phi, rng)
            est = estimate_delay_to_memory(x, tau=10)
            truth = memory_length_ar1(phi)
            assert 0.5 * truth <= est.memory_length <= 2.0 * truth
            assert est.ratio == pytest.approx(10.0 / est.memory_length)
            assert est.decay_per_step == pytest.approx(np.exp(-1.0 / est.memory_length))
            assert est.lags_used >= 2
            lengths[phi] = est.memory_length
        assert lengths[0.6] < lengths[0.85] < lengths[0.95]  # orders persistence
        # long-memory case is accurate to ~15%
        assert lengths[0.95] == pytest.approx(memory_length_ar1(0.95), rel=0.15)


def test_estimate_delay_to_memory_garch_squares() -> None:
    """SYNTHETIC: feature='square' targets L_vol decay of squared residuals."""
    for seed in (77, 78):
        lengths: dict[float, float] = {}
        for ab in (0.9, 0.95, 0.99):
            rng = np.random.default_rng(seed)
            x = _garch(12000, 0.1, ab - 0.1, rng)
            est = estimate_delay_to_memory(x, tau=10, feature="square", max_lags=10)
            truth = memory_length_garch(0.1, ab - 0.1)
            assert 0.5 * truth <= est.memory_length <= 2.0 * truth
            lengths[ab] = est.memory_length
        assert lengths[0.9] < lengths[0.95] < lengths[0.99]


def test_estimate_delay_to_memory_recovers_drift_memory() -> None:
    """SYNTHETIC: on AR(1)-drift + noise, the fitted RATE sees through the
    signal-plus-noise attenuation (constant factor -> intercept, not slope)."""
    eps = _drift_stream(6000, 0.98, 17)
    est = estimate_delay_to_memory(eps, tau=10)
    truth = memory_length_ar1(0.98)
    assert est.memory_length == pytest.approx(truth, rel=0.15)
    assert est.ratio == pytest.approx(10.0 / truth, rel=0.15)


@pytest.mark.parametrize("seed", [11, 12, 13, 4242])
def test_estimate_delay_to_memory_white_noise_fails_closed(seed: int) -> None:
    """No memory scale is fabricated for (near-)white streams."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 6000)
    with pytest.raises(ValueError, match="decay"):
        estimate_delay_to_memory(x, tau=5)


def test_estimate_delay_to_memory_input_edges_fail_closed() -> None:
    rng = np.random.default_rng(4)
    good = _ar1(500, 0.8, rng)
    with pytest.raises(ValueError, match="1-d"):
        estimate_delay_to_memory(good.reshape(10, 50), tau=1)
    with pytest.raises(ValueError, match="need at least"):
        estimate_delay_to_memory(good[:50], tau=1)  # < 10 * max_lags
    with pytest.raises(ValueError, match="constant"):
        estimate_delay_to_memory(np.zeros(500), tau=1)
    with pytest.raises(ValueError, match="feature"):
        estimate_delay_to_memory(good, tau=1, feature="abs")
    with pytest.raises(ValueError, match="max_lags"):
        estimate_delay_to_memory(good, tau=1, max_lags=1)
    with pytest.raises(ValueError, match="acf_floor"):
        estimate_delay_to_memory(good, tau=1, acf_floor=0.0)
    with pytest.raises(ValueError, match="tau"):
        estimate_delay_to_memory(good, tau=0)
    with pytest.raises(ValueError, match="finite"):
        estimate_delay_to_memory(np.full(500, np.nan), tau=1)


# ------------------------------------------------------------- metrics
def test_worst_case_local_coverage_error() -> None:
    err = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    # window means over k=4: 0, .25, .5, .75, 1, .75, .5, .25, 0 -> max |0.1 - m| = 0.9
    assert worst_case_local_coverage_error(err, ALPHA, 4) == pytest.approx(0.9)
    assert worst_case_local_coverage_error(np.zeros(50), ALPHA, 10) == pytest.approx(ALPHA)
    assert worst_case_local_coverage_error(np.ones(50), ALPHA, 10) == pytest.approx(1 - ALPHA)
    with pytest.raises(ValueError, match="k"):
        worst_case_local_coverage_error(err, ALPHA, 0)
    with pytest.raises(ValueError, match="k"):
        worst_case_local_coverage_error(err, ALPHA, 13)
    with pytest.raises(ValueError, match="err"):
        worst_case_local_coverage_error(np.full(10, 1.5), ALPHA, 5)
    with pytest.raises(ValueError, match="1-d"):
        worst_case_local_coverage_error(err.reshape(3, 4), ALPHA, 4)


def test_adaptation_duration_first_dip_and_recovery() -> None:
    covered = np.concatenate([np.ones(300), np.zeros(300), np.ones(300)])
    dur, recovered = adaptation_duration(covered, threshold=0.89, roll=100)
    # dip at rolling start 212 (12 zeros in window), recovery at 589 (11 zeros)
    assert (dur, recovered) == (377, True)
    censored = np.concatenate([np.ones(300), np.zeros(300)])
    dur_c, rec_c = adaptation_duration(censored, threshold=0.89, roll=100)
    assert (dur_c, rec_c) == (501 - 212, False)  # right-censored at series end
    assert adaptation_duration(np.ones(400)) == (0, True)  # no dip at all
    with pytest.raises(ValueError, match="threshold"):
        adaptation_duration(covered, threshold=1.5)
    with pytest.raises(ValueError, match="roll"):
        adaptation_duration(covered, roll=0)
    with pytest.raises(ValueError, match="search_start"):
        adaptation_duration(covered, search_start=10_000)
    with pytest.raises(ValueError, match="covered"):
        adaptation_duration(np.full(300, 0.5), roll=100)
    with pytest.raises(ValueError, match="1-d"):
        adaptation_duration(covered.reshape(-1, 1), roll=100)


def test_curve_collapse_reduction_synthetic_and_edges() -> None:
    """A value that is a pure function of r collapses under r, not under τ."""
    lengths = (2.0, 40.0)
    taus = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    ratios, delays, values = [], [], []
    for length in lengths:
        for tau in taus:
            r = tau / length
            ratios.append(r)
            delays.append(tau)
            values.append(1.0 / (1.0 + r))
    red = curve_collapse_reduction(
        np.asarray(values), np.asarray(ratios), np.asarray(delays), n_bins=12
    )
    assert red > 50.0  # measured ~90%: r is the organizing axis by construction
    with pytest.raises(ValueError, match="equal length"):
        curve_collapse_reduction(np.ones(4), np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="positive"):
        curve_collapse_reduction(np.ones(4), -np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="n_bins"):
        curve_collapse_reduction(np.ones(4), np.ones(4), np.ones(4), n_bins=1)
    spread = np.asarray([1.0, 100.0, 10_000.0])
    with pytest.raises(ValueError, match="no bin"):
        curve_collapse_reduction(np.arange(3.0), spread, spread, n_bins=12)


# ------------------------------------------- run_delayed_conformal mechanics
def test_run_delayed_conformal_mechanics_and_determinism() -> None:
    rng = np.random.default_rng(8)
    eps = _ar1(600, 0.7, rng)
    tau, window, gamma = 4, 100, 0.02
    res = run_delayed_conformal(eps, tau, alpha=ALPHA, gamma=gamma, window=window)
    n_c = eps.size - window - tau + 1
    assert n_c == 497
    for arr in (res.levels, res.lower, res.upper, res.covered, res.err, res.interval_score):
        assert arr.shape == (n_c,)
    assert np.all((res.covered == 0.0) | (res.covered == 1.0))
    assert np.array_equal(res.err, 1.0 - res.covered)
    assert np.array_equal(res.targets, eps[window - 1 + tau : window - 1 + tau + n_c])
    assert np.array_equal(res.levels, res.controller.alpha_history)
    # interval score is the paper's Eq. (15), recomputed independently
    iscore = (
        (res.upper - res.lower)
        + (2.0 / ALPHA) * np.maximum(res.lower - res.targets, 0.0)
        + (2.0 / ALPHA) * np.maximum(res.targets - res.upper, 0.0)
    )
    assert np.array_equal(res.interval_score, iscore)
    # delay alignment: the feedback consumed at construction c is exactly the
    # outcome of construction c - tau (target revealed now), NaN before that
    fb = res.controller.feedback_history
    assert np.all(np.isnan(fb[:tau]))
    assert np.array_equal(fb[tau:], res.err[: n_c - tau])
    # bands are the equal-tail window quantiles at the issued level
    assert np.all(res.lower < res.upper)
    assert res.empirical_coverage == pytest.approx(float(np.mean(res.covered)))
    assert res.long_run_deviation == pytest.approx(abs(float(np.mean(res.err)) - ALPHA))
    assert res.coverage_bound == pytest.approx(long_run_coverage_bound(ALPHA, gamma, tau, n_c))
    again = run_delayed_conformal(eps, tau, alpha=ALPHA, gamma=gamma, window=window)
    assert np.array_equal(again.levels, res.levels)
    assert np.array_equal(again.covered, res.covered)


def test_run_delayed_conformal_fail_closed() -> None:
    rng = np.random.default_rng(9)
    eps = rng.normal(0.0, 1.0, 200)
    with pytest.raises(ValueError, match="1-d"):
        run_delayed_conformal(eps.reshape(20, 10), 1, window=50)
    with pytest.raises(ValueError, match="at least"):
        run_delayed_conformal(eps[:60], 5, window=60)
    with pytest.raises(ValueError, match="window"):
        run_delayed_conformal(eps, 5, window=1)
    with pytest.raises(ValueError, match="window"):
        run_delayed_conformal(eps, 5, window=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        run_delayed_conformal(np.full(200, np.nan), 5, window=50)


# ------------------------------------- MC: AR(1)-drift organizes by r (SYNTHETIC)
def test_ar1_drift_ratio_orders_coverage_deviation(
    ar1_drift_sweep: tuple[list[dict[str, float]], float],
) -> None:
    """SYNTHETIC: at fixed memory L, the coverage deviation is ordered by r ∝ τ.

    Sweeping τ ∈ {1, 6, 18} at each φ ∈ {0.7, 0.9, 0.98} (L ≈ 2.80, 9.49,
    49.50; r ≈ 0.02..6.4), the RMS deviation of rolling-100 coverage from the
    nominal 0.9 increases strictly with r within every persistence level —
    the horizon-mismatch mechanism of the paper's marginal bound (Eq. (12):
    E|α*_{t+τ} − α*_{t}| grows with τ relative to memory). Pooled ordering
    across φ by r alone is NOT asserted: the paper itself notes local
    coverage-error curves do not collapse across persistence levels (Sec.
    7.2) — the interval-score collapse is the pooled claim, tested next.
    """
    rows, _ = ar1_drift_sweep
    for phi in SWEEP_PHIS:
        devs = [x["dev"] for x in rows if x["phi"] == phi]
        assert devs[0] < devs[1] < devs[2], f"phi={phi}: {devs}"
        iscs = [x["isc"] for x in rows if x["phi"] == phi]
        assert iscs[0] < iscs[1] < iscs[2], f"phi={phi}: {iscs}"


def test_ar1_drift_interval_score_collapses_under_ratio(
    ar1_drift_sweep: tuple[list[dict[str, float]], float],
) -> None:
    """SYNTHETIC: pooled interval scores align tighter under r = τ/L than τ.

    The paper's curve-collapse diagnostic (Eq. (17), Sec. 7.2: ≈79% within-bin
    scatter reduction at their scale with γ* selection); this smaller seeded
    sweep reproduces the direction and a substantial reduction. Descriptive —
    a collapse, not a guarantee.
    """
    rows, _ = ar1_drift_sweep
    values = np.array([x["isc"] for x in rows])
    ratio_axis = np.array([x["r"] for x in rows])
    delay_axis = np.array([x["tau"] for x in rows])
    red = curve_collapse_reduction(values, ratio_axis, delay_axis, n_bins=4)
    assert red > 25.0, f"scatter reduction under r: {red:.1f}%"


def test_ar1_drift_transition_near_ratio_one(
    ar1_drift_sweep: tuple[list[dict[str, float]], float],
) -> None:
    """SYNTHETIC: efficiency beats the exchangeable baseline for r ≲ 1 and
    degrades to/over it for r ≳ 1 (paper Sec. 7.2: transition near r ≈ 1)."""
    rows, baseline = ar1_drift_sweep
    low = [x["isc"] for x in rows if x["r"] < 0.4]
    high = [x["isc"] for x in rows if x["r"] > 1.5]
    assert low and high
    assert float(np.mean(low)) < baseline
    assert float(np.mean(high)) > baseline
    smallest_r = min(rows, key=lambda x: x["r"])  # phi=0.98, tau=1, r≈0.02
    assert smallest_r["isc"] < baseline


# --------------------------------- MC: abrupt variance shift vs iid (SYNTHETIC)
VAR_SHIFT_GAMMAS = (0.002, 0.128)


def _variance_shift_runs(gamma: float) -> tuple[list[float], list[bool], float]:
    durs, recs, ratios = [], [], []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(500 + seed)
        eps = np.concatenate([rng.normal(0.0, 1.0, 1200), rng.normal(0.0, 4.0, 1200)])
        res = run_delayed_conformal(eps, 10, alpha=ALPHA, gamma=gamma, window=250)
        start = max(0, 1200 - 250 + 1 - 100)
        dur, rec = adaptation_duration(res.covered, 0.89, 100, search_start=start)
        durs.append(float(dur))
        recs.append(rec)
        widths = res.upper - res.lower
        ratios.append(float(np.mean(widths[-500:]) / np.mean(widths[:500])))
    return durs, recs, float(np.mean(ratios))


def test_variance_shift_prefers_larger_gamma() -> None:
    """SYNTHETIC: after an abrupt σ 1→4 shift the LARGE step size recovers
    coverage faster (paper Sec. 7.4/8: severe shifts favor larger γ*; R=100's
    smaller selected γ* produced longer dips for variance shifts)."""
    durs_small, rec_small, ratio_small = _variance_shift_runs(VAR_SHIFT_GAMMAS[0])
    durs_large, rec_large, _ = _variance_shift_runs(VAR_SHIFT_GAMMAS[1])
    assert all(rec_small) and all(rec_large)
    assert float(np.mean(durs_large)) < float(np.mean(durs_small))
    # sustained regime-specific width plateau: post-shift bands widen toward
    # the new scale (paper Sec. 7.4), not a transient spike
    assert ratio_small > 2.0


def test_iid_prefers_smaller_gamma_without_coverage_benefit() -> None:
    """SYNTHETIC: under exchangeable residuals larger γ monotonically worsens
    the interval score with no coverage benefit (paper Sec. 7.1) — the
    adaptation-rate preference flips versus the variance-shift stream."""
    means, covs = [], []
    for gamma in (0.002, 0.008, 0.032, 0.128):
        iscs, cvg = [], []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(700 + seed)
            eps = rng.normal(0.0, 1.0, 2400)
            res = run_delayed_conformal(eps, 10, alpha=ALPHA, gamma=gamma, window=250)
            iscs.append(res.mean_interval_score)
            cvg.append(res.empirical_coverage)
        means.append(float(np.mean(iscs)))
        covs.append(float(np.mean(cvg)))
    assert means[0] < means[1] < means[2] < means[3]
    assert all(abs(c - NOMINAL) <= 0.02 for c in covs)


# ---------------------------------------------------------------- honesty
def test_module_has_no_forbidden_metrics() -> None:
    import quant_fund.models.delayed_aci as delayed_aci

    text = pathlib.Path(delayed_aci.__file__).read_text(encoding="utf-8").lower()
    for token in ("sharpe", "sortino", "calmar", "p&l", "pnl", "nav"):
        assert token not in text


def test_docstring_documents_honesty_and_citation() -> None:
    """The module docstring must carry the verified citation and the honesty
    caveats: worst-case bounds, descriptive r-diagnostic, SYNTHETIC-only."""
    import quant_fund.models.delayed_aci as delayed_aci

    doc = (delayed_aci.__doc__ or "").lower()
    assert "arxiv:2609.07251" in doc
    assert "el halabi" in doc
    assert "worst-case" in doc
    assert "descriptive" in doc
    assert "synthetic" in doc
    # tau = 1 reduction to Gibbs & Candes ACI is documented
    assert "gibbs" in doc and "2106.00170" in doc


def test_public_api_surface() -> None:
    import quant_fund.models.delayed_aci as delayed_aci

    expected = {
        "DelayedACI",
        "DelayedConformalResult",
        "DelayToMemoryEstimate",
        "adaptation_duration",
        "curve_collapse_reduction",
        "delay_to_memory_ratio",
        "estimate_delay_to_memory",
        "long_run_coverage_bound",
        "marginal_coverage_bound",
        "memory_length_ar1",
        "memory_length_garch",
        "memory_length_markov",
        "optimal_gamma",
        "run_delayed_conformal",
        "worst_case_local_coverage_error",
    }
    assert set(delayed_aci.__all__) == expected
    for name in delayed_aci.__all__:
        assert hasattr(delayed_aci, name)
