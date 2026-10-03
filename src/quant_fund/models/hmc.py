"""Hamiltonian Monte Carlo + NUTS sampling.

Leapfrog-integrated HMC with dual-averaging step-size adaptation
(Nesterov/Hoffman-Gelman) and the multinomial NUTS sampler (no U-turn;
recursive doubling terminated on U-turn or divergence). Operates on an
arbitrary ``logp_grad(x) -> (logp, grad)`` oracle — no autodiff
dependency.

References
----------
- Neal (2011). MCMC using Hamiltonian dynamics. *Handbook of Markov
  Chain Monte Carlo*, ch. 5.
- Hoffman & Gelman (2014). The No-U-Turn sampler. *JMLR* 15:1593–1623 —
  arXiv:1111.4246.
- Betancourt (2017). A conceptual introduction to Hamiltonian Monte
  Carlo — arXiv:1701.02434 (multinomial sampling variant).

Honesty
-------
All evaluations are SYNTHETIC targets (correlated Gaussian, Neal's
funnel, Student-t). Reported keys are sampler diagnostics — effective
sample size, R-hat, energy error, funnel capture — never parameter
estimates of real market quantities.

Composition notes
-----------------
- ``models/pmcmc_sv.py``: particle MCMC for state-space likelihoods.
  This module is a generic continuous-space sampler — orthogonal
  machinery.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LogpGrad = Callable[[FloatArray], tuple[float, FloatArray]]


def _as_state(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("state must be non-empty and finite")
    return x


def _call(f: LogpGrad, x: FloatArray) -> tuple[float, FloatArray]:
    lp, g = f(x)
    g = np.asarray(g, dtype=np.float64).ravel()
    if g.size != x.size:
        raise ValueError("gradient size mismatch")
    return float(lp), g


def leapfrog(
    logp_grad: LogpGrad, x: FloatArray, p: FloatArray, eps: float, steps: int
) -> tuple[FloatArray, FloatArray]:
    """Leapfrog trajectory: half-step momentum, full-step position
    interleaved. Returns (x_new, p_new) or raises on divergence."""
    x = x.copy()
    p = p.copy()
    _lp, g = _call(logp_grad, x)
    p += 0.5 * eps * g
    for i in range(steps):
        x += eps * p
        _lp, g = _call(logp_grad, x)
        if not np.all(np.isfinite(x)) or bool(np.abs(x).max() > 1e150):
            raise FloatingPointError("leapfrog diverged")
        p += eps * g if i < steps - 1 else 0.5 * eps * g
    return x, p


def hmc_sample(
    logp_grad: LogpGrad,
    x0: FloatArray,
    eps: float = 0.1,
    n_leap: int = 10,
    n_draws: int = 100,
    burn: int = 50,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Static-HMC chain. Returns (draws, accept_rate per iter)."""
    x = _as_state(x0)
    if not (eps > 0 and n_leap >= 1 and n_draws >= 1 and burn >= 0):
        raise ValueError("invalid sampler sizes")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, x.size))
    accepts = np.zeros(n_draws)
    for i in range(n_draws + burn):
        p0 = rng.standard_normal(x.size)
        lp0, _g = _call(logp_grad, x)
        h0 = -lp0 + 0.5 * p0 @ p0
        try:
            x1, p1 = leapfrog(logp_grad, x, p0, eps, n_leap)
            lp1, _ = _call(logp_grad, x1)
            h1 = -lp1 + 0.5 * p1 @ p1
            u = math.log(rng.uniform())
            if u < h0 - h1:
                x = x1
                if i >= burn:
                    accepts[i - burn] = 1.0
        except FloatingPointError:
            pass
        if i >= burn:
            draws[i - burn] = x
    return draws, accepts


class DualAverage:
    """Nesterov dual-averaging step-size adapter (Hoffman-Gelman §3.2.1).

    ``step(accept, t)`` returns the current primal eps; after warmup use
    ``final_eps`` (the smoothed ``eps_bar``) rather than the noisy
    instantaneous value.
    """

    def __init__(self, eps0: float, target_accept: float = 0.65) -> None:
        if not (eps0 > 0 and 0 < target_accept < 1):
            raise ValueError("invalid dual-averaging config")
        self._mu = math.log(10.0 * eps0)
        self._log_eps_bar = math.log(eps0)
        self._h_bar = 0.0
        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75
        self.target = target_accept

    def step(self, accept: float, t: int) -> float:
        self._h_bar = (1 - 1 / (t + self._t0)) * self._h_bar + (self.target - accept) / (
            t + self._t0
        )
        log_eps = self._mu - math.sqrt(t) / self._gamma * self._h_bar
        eta = t**-self._kappa
        self._log_eps_bar = eta * log_eps + (1 - eta) * self._log_eps_bar
        return math.exp(log_eps)

    @property
    def final_eps(self) -> float:
        return math.exp(self._log_eps_bar)


def _uturn(
    x_minus: FloatArray, x_plus: FloatArray, p_minus: FloatArray, p_plus: FloatArray
) -> bool:
    """U-turn check on the doubling trajectory (Betancourt criterion)."""
    dx = x_plus - x_minus
    return bool(dx @ p_minus < 0 or dx @ p_plus < 0)


def nuts_sample(
    logp_grad: LogpGrad,
    x0: FloatArray,
    n_draws: int = 100,
    burn: int = 50,
    eps: float = 0.1,
    max_depth: int = 8,
    target_accept: float = 0.7,
    adapt: bool = True,
    eps_max: float = math.inf,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Multinomial NUTS (Hoffman-Gelman 2014 + Betancourt 2017).

    Recursive doubling, uniform proposal over in-slice states, dual-
    averaged step size frozen after warmup at the smoothed eps_bar
    (clamped by ``eps_max``). Returns (draws, tree_depths, energy_errs).
    """
    x = _as_state(x0)
    if n_draws < 1 or eps <= 0 or max_depth < 1:
        raise ValueError("invalid NUTS config")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, x.size))
    depths = np.zeros(n_draws)
    e_errs = np.zeros(n_draws)
    adapter = DualAverage(eps, target_accept) if adapt else None
    for i in range(n_draws + burn):
        p0 = rng.standard_normal(x.size)
        lp0, _ = _call(logp_grad, x)
        h0 = float(-lp0 - 0.5 * p0 @ p0)
        # slice variable: logu = -H0 + log(U'), U' ~ U(0,1);
        # acceptable states satisfy h < -logu.
        logu = -h0 + math.log(rng.uniform())
        x_minus = x_plus = x.copy()
        p_minus = p_plus = p0.copy()
        # Hoffman-Gelman slice-sampling NUTS: candidates are all states
        # visited inside the slice; the proposal is uniform over them.
        candidates: list[FloatArray] = []
        depth_used = 0
        accept_stat = 0.0
        n_stat = 0
        for depth in range(max_depth):
            direction = rng.choice([-1, 1])
            n_leap = 2**depth
            x_sub = x_minus if direction < 0 else x_plus
            p_sub = p_minus if direction < 0 else p_plus
            n_new: list[FloatArray] = []
            diverged = False
            for _ in range(n_leap):
                try:
                    x_sub, p_sub = leapfrog(logp_grad, x_sub, p_sub, direction * eps, 1)
                except FloatingPointError:
                    # leapfrog blow-up: reject for adaptation
                    n_stat += 1
                    diverged = True
                    break
                lp_s, _g2 = _call(logp_grad, x_sub)
                with np.errstate(invalid="ignore", over="ignore"):
                    h_s = float(-lp_s - 0.5 * p_sub @ p_sub)
                if not math.isfinite(h_s) or -h_s < logu - 1000.0:
                    # divergence counts as a rejection for dual averaging
                    n_stat += 1
                    diverged = True
                    break
                # per-state Metropolis accept feeds dual averaging
                accept_stat += min(1.0, math.exp(max(-700.0, min(700.0, h0 - h_s))))
                n_stat += 1
                if -h_s > logu:
                    n_new.append(x_sub.copy())
            if direction < 0:
                x_minus, p_minus = x_sub, p_sub
            else:
                x_plus, p_plus = x_sub, p_sub
            candidates.extend(n_new)
            if diverged or _uturn(x_minus, x_plus, p_minus, p_plus):
                break
            depth_used = depth + 1
        # uniform proposal over the slice (incl. current position)
        if candidates and rng.uniform() < len(candidates) / (len(candidates) + 1):
            x = candidates[int(rng.integers(len(candidates)))]
        lp1, _ = _call(logp_grad, x)
        h1 = float(-lp1 - 0.5 * p0 @ p0)
        if i >= burn:
            draws[i - burn] = x
            depths[i - burn] = depth_used
            # energy-error proxy: |ΔH| measured on the shared momentum
            e_errs[i - burn] = abs(h1 - h0)
        elif adapter is not None:
            eps = min(adapter.step(accept_stat / max(n_stat, 1), i + 1), eps_max)
            if i == burn - 1:
                eps = min(adapter.final_eps, eps_max)
    return draws, depths, e_errs


def ess(x: FloatArray, max_lag: int | None = None) -> float:
    """Effective sample size via the initial-positive sequence."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = float(x @ x / n)
    if var <= 0:
        return 1.0
    max_lag = min(max_lag or n // 2, n - 1)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        ac = float(x[:-lag] @ x[lag:] / (n - lag)) / var
        if ac <= 0:
            break
        rho_sum += ac
    return float(n / (1.0 + 2.0 * rho_sum))


def rhat(chains: FloatArray) -> float:
    """Potential scale reduction factor across parallel chains (rows)."""
    ch = np.asarray(chains, dtype=np.float64)
    if ch.ndim != 2 or ch.shape[0] < 2 or ch.shape[1] < 4:
        raise ValueError("chains must be (n_chains>=2, n_draws>=4)")
    m, n = ch.shape
    means = ch.mean(axis=1)
    var_in = ch.var(axis=1, ddof=1).mean()
    var_b = n * means.var(ddof=1)
    var_hat = (n - 1) / n * var_in + var_b / n
    return float(math.sqrt(max(var_hat / max(var_in, 1e-300), 1e-300)))


# --- synthetic targets ------------------------------------------------------


def synth_gaussian(dim: int = 3, corr: float = 0.8, seed: int = 0) -> LogpGrad:
    """Correlated-Gaussian target; returns logp_grad oracle."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim))
    cov = np.eye(dim) * (1 - corr) + corr
    cov = a @ a.T * 0.1 + cov
    prec = np.linalg.inv(cov)

    def f(x: FloatArray) -> tuple[float, FloatArray]:
        return float(-0.5 * x @ prec @ x), np.asarray(-prec @ x, dtype=np.float64)

    return f


def synth_funnel(dim: int = 4, seed: int = 0) -> LogpGrad:
    """Neal's funnel: y ~ N(0, 9); x_j | y ~ N(0, exp(y))."""

    def f(x: FloatArray) -> tuple[float, FloatArray]:
        y = float(x[0])
        rest = x[1:]
        g = np.empty(x.size)
        if abs(y) > 1e100:
            # saturated tail: keep finite gradient pulling back to 0
            g[0] = -math.copysign(1e100, y)
            g[1:] = -rest * 1e-300
            return -1e300, g
        # variance exp(y) is clamped for numerical safety — regions
        # y > 50 are already killed by the -y^2/18 Gaussian term, and
        # exp(y) underflow for very negative y is floored.
        ey = math.exp(min(y, 50.0)) if y > -300.0 else 1e-300
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            lp = -0.5 * y**2 / 9.0 - 0.5 * rest.size * y - 0.5 * float(rest @ rest) / ey
            g[0] = (
                -y / 9.0 - 0.5 * rest.size + (0.5 * float(rest @ rest) / ey if y <= 50.0 else 0.0)
            )
            g[1:] = -rest / ey
        if not np.all(np.isfinite(g)):
            g[~np.isfinite(g)] = 0.0
        return lp, g

    return f


def synth_student_t(df: float = 4.0, dim: int = 2, seed: int = 0) -> LogpGrad:
    """Correlated Student-t target."""
    cov = np.array([[1.0, 0.7], [0.7, 1.5]])[:dim, :dim]
    prec = np.linalg.inv(cov)

    def f(x: FloatArray) -> tuple[float, FloatArray]:
        q = float(x @ prec @ x)
        lp = -0.5 * (df + x.size) * math.log1p(q / df)
        g = -(df + x.size) / (df + q) * (prec @ x)
        return lp, g

    return f


def bench_hmc(seed: int = 20261231 + 161) -> dict[str, float]:
    """SYNTHETIC HMC/NUTS sampler diagnostics."""
    g = synth_gaussian(dim=3, corr=0.8, seed=seed)
    # HMC chain: ESS per draw, accept rate
    d1, acc = hmc_sample(g, np.zeros(3), eps=0.15, n_leap=15, n_draws=800, burn=200, seed=seed)
    ess_min = min(ess(d1[:, j]) for j in range(3))
    # NUTS: same target
    dn, depths, e_errs = nuts_sample(
        g,
        np.zeros(3),
        n_draws=500,
        burn=200,
        eps=0.3,
        adapt=False,
        seed=seed + 1,
    )
    ess_n = min(ess(dn[:, j]) for j in range(3))
    # funnel capture: does NUTS reach the narrow neck (y < -3)?
    funnel = synth_funnel(dim=3, seed=seed)
    df_, _dep, _ee = nuts_sample(
        funnel,
        np.array([0.0, 0.0, 0.0]),
        n_draws=800,
        burn=200,
        eps=0.2,
        max_depth=7,
        target_accept=0.85,
        eps_max=0.8,
        seed=seed + 2,
    )
    funnel_cap = float(np.mean(df_[:, 0] < -3.0))
    # R-hat across 4 short chains on the Gaussian
    chains = np.stack(
        [
            nuts_sample(
                g,
                np.zeros(3) + s,
                n_draws=200,
                burn=200,
                eps=0.3,
                adapt=False,
                seed=seed + 10 + s,
            )[0][:, 0]
            for s in range(4)
        ]
    )
    rh = rhat(chains)
    det = float(
        np.array_equal(
            hmc_sample(g, np.zeros(3), eps=0.15, n_leap=15, n_draws=20, burn=5, seed=9)[0],
            hmc_sample(g, np.zeros(3), eps=0.15, n_leap=15, n_draws=20, burn=5, seed=9)[0],
        )
    )
    return {
        "synthetic_hmc_ess_per_draw": ess_min / d1.shape[0],
        "synthetic_hmc_accept": float(acc.mean()),
        "synthetic_nuts_ess_per_draw": ess_n / dn.shape[0],
        "synthetic_nuts_mean_depth": float(depths.mean()),
        "synthetic_energy_err_mean": float(np.mean(np.abs(e_errs))),
        "synthetic_funnel_capture": funnel_cap,
        "synthetic_rhat": rh,
        "synthetic_determinism": det,
    }
