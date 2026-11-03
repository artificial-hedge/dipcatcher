"""Particle MCMC (PMMH) parameter inference for stochastic volatility.

Andrieu-Doucet-Holenstein pseudo-marginal Metropolis-Hastings: the intractable
SV marginal likelihood is replaced at every MH step by an unbiased particle-
filter estimate — composing ``models.nonlinear_filters.particle_filter``
directly (AR(1) drift as ``f``, Gaussianization as ``q_std``, log-vol
Gaussianization of returns as ``obs_loglik``). The PMMH chain then targets
the exact joint posterior of (mu, phi, sigma_eta) — PF variance inflates
acceptance noise but not the invariant distribution.

SV DGP (Kim-Shephard-Chib style, AR(1) log-vol):

    h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eps_t
    r_t = exp(h_t / 2) * e_t

References
----------
- Andrieu, Doucet & Holenstein (2010). Particle Markov chain Monte Carlo
  methods. *JRSS-B* 72(3):269-342 (journal; the PMMH theorem).
- Kim, Shephard & Chib (1998). Stochastic volatility: likelihood inference
  and comparison with ARCH models. *Review of Economic Studies* 65:361-393
  (journal; the AR(1) SV model).
- Pitt, Silva, Giordani & Kohn (2012). On some properties of Markov chain
  Monte Carlo simulation methods based on the particle filter.
  *Journal of Econometrics* 171(2):134-151 (journal; PF-variance guidance).

Honesty
-------
All benches run on seeded SYNTHETIC SV paths generated in-module; recovery
numbers validate the machinery only — never market evidence.

Composition notes
-----------------
- ``models/nonlinear_filters.py``: the marginal likelihood is the same
  bootstrap SIR recursion as ``particle_filter`` (AR(1) transition,
  Gaussian observation, systematic resampling). The step is a vectorized
  kernel so the PMMH chain does not pay a Python call per particle.
- ``models/stoch_vol.py``: closed-form/approximate SV estimation; PMCMC is
  the fully-Bayesian counterpart.
- ``models/bayesian.py``: conjugate/Gibbs machinery; PMMH here is
  likelihood-free via particles instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import exp, isfinite, log, pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a required dependency

    def njit(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _deco(fn: Any) -> Any:
            return fn

        if len(_args) == 1 and callable(_args[0]) and not _kwargs:
            return _args[0]
        return _deco


FloatArray = NDArray[np.float64]

_Log2Pi = log(2.0 * pi)


@njit(cache=True)
def _sv_bootstrap_loglik(
    y2: np.ndarray,
    mu: float,
    phi: float,
    sigma_eta: float,
    parts: np.ndarray,
    noise: np.ndarray,
    uniforms: np.ndarray,
    log2pi: float,
) -> float:
    """Bootstrap SIR marginal log-likelihood for the AR(1) SV model.

    ``noise`` and ``uniforms`` are drawn by the caller from ``default_rng``
    so the estimate stays deterministic for a fixed seed. Weights follow the
    Gordon–Salmond–Smith recursion: incremental loglik is
    ``log(sum_i w_{t-1,i} p(y_t | x_t,i))``, with systematic resampling when
    ESS < N/2.
    """
    n = y2.shape[0]
    n_particles = parts.shape[0]
    inv_n = 1.0 / n_particles
    w = np.empty(n_particles, dtype=np.float64)
    work = np.empty(n_particles, dtype=np.float64)
    chosen = np.empty(n_particles, dtype=np.float64)
    for i in range(n_particles):
        w[i] = inv_n
    loglik = 0.0
    thresh = 0.5 * n_particles
    one_minus_phi = 1.0 - phi
    for t in range(n):
        mx = -1.0e300
        y2_t = y2[t]
        for i in range(n_particles):
            h = mu * one_minus_phi + phi * parts[i] + sigma_eta * noise[t, i]
            parts[i] = h
            ll = -0.5 * (log2pi + h + y2_t * math.exp(-h))
            work[i] = ll
            if ll > mx:
                mx = ll
        if mx != mx:  # NaN
            return np.nan
        tot = 0.0
        for i in range(n_particles):
            wi = w[i] * math.exp(work[i] - mx)
            w[i] = wi
            tot += wi
        if not (tot > 0.0) or tot != tot:
            return np.nan
        inv_tot = 1.0 / tot
        ess_acc = 0.0
        for i in range(n_particles):
            w[i] *= inv_tot
            ess_acc += w[i] * w[i]
        loglik += mx + math.log(tot)
        if ess_acc > 0.0 and (1.0 / ess_acc) < thresh:
            c = 0.0
            for i in range(n_particles):
                c += w[i]
                work[i] = c
            j = 0
            u0 = uniforms[t] * inv_n
            last = n_particles - 1
            for i in range(n_particles):
                u = u0 + i * inv_n
                while j < last and work[j] < u:
                    j += 1
                chosen[i] = parts[j]
            for i in range(n_particles):
                parts[i] = chosen[i]
                w[i] = inv_n
    return loglik


def _check_returns(r: FloatArray, min_len: int = 20) -> FloatArray:
    v = np.asarray(r, dtype=float).ravel()
    if v.size < min_len:
        raise ValueError(f"returns need >= {min_len} points")
    if not np.isfinite(v).all():
        raise ValueError("returns contain non-finite values")
    return v


def simulate_sv(
    n: int,
    mu: float = -4.0,
    phi: float = 0.95,
    sigma_eta: float = 0.15,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Simulate the AR(1)-log-vol SV model. Returns (returns, log_vols)."""
    if n < 8:
        raise ValueError("n >= 8 required")
    if abs(phi) >= 1.0 or sigma_eta <= 0:
        raise ValueError("need |phi|<1 and sigma_eta>0")
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = mu + sigma_eta / sqrt(1 - phi * phi) * rng.standard_normal()
    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.standard_normal()
    r = np.exp(h / 2.0) * rng.standard_normal(n)
    return r, h


def sv_pf_loglik(
    returns: FloatArray,
    mu: float,
    phi: float,
    sigma_eta: float,
    n_particles: int = 300,
    seed: int = 0,
) -> float:
    """Unbiased-ish marginal loglik estimate via a bootstrap particle filter.

    Same SIR recursion as ``nonlinear_filters.particle_filter`` for this
    Gaussian SV model, evaluated in a vectorized kernel.
    """
    r = _check_returns(returns)
    if abs(phi) >= 1.0 or sigma_eta <= 0:
        raise ValueError("need |phi|<1 and sigma_eta>0")
    if not 10 <= n_particles <= 100_000:
        raise ValueError("n_particles must be in [10, 100000]")

    p0 = sigma_eta / sqrt(max(1.0 - phi * phi, 1e-4))
    rng = np.random.default_rng(seed)
    parts = np.ascontiguousarray(rng.normal(mu, p0, n_particles), dtype=np.float64)
    noise = np.ascontiguousarray(rng.standard_normal((r.size, n_particles)), dtype=np.float64)
    uniforms = np.ascontiguousarray(rng.random(r.size), dtype=np.float64)
    out = _sv_bootstrap_loglik(
        np.ascontiguousarray(r * r, dtype=np.float64),
        float(mu),
        float(phi),
        float(sigma_eta),
        parts,
        noise,
        uniforms,
        _Log2Pi,
    )
    if not isfinite(out):
        raise ValueError("particle degeneracy: non-finite likelihood")
    return float(out)


@dataclass(frozen=True)
class PMMHResult:
    """Posterior summary from the PMMH chain."""

    post_mean: FloatArray  # (mu, phi, sigma_eta)
    post_std: FloatArray
    acceptance_rate: float
    n_iter: int
    chain_loglik: FloatArray  # accepted-path estimated loglik
    effective_sizes: FloatArray  # per-param ESS estimate


def _ess_1d(x: FloatArray) -> float:
    """Initial-monotone-sequence ESS estimate."""
    n = x.size
    if n < 8:
        return float(n)
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return float(n)
    rho_prev = 1.0
    ess = float(n)
    for lag in range(1, min(n // 4, 200)):
        rho = float(x[:-lag] @ x[lag:]) / denom
        pair = rho_prev + rho
        if pair <= 0:
            break
        ess = float(n) / (1.0 + 2.0 * pair * (lag if lag > 1 else 1) / 2.0)
        rho_prev = rho
    return max(ess, 1.0)


def pmmh_sv(
    returns: FloatArray,
    n_iter: int = 400,
    n_particles: int = 300,
    theta0: FloatArray | None = None,
    proposal_sd: FloatArray | None = None,
    burnin_frac: float = 0.5,
    seed: int = 0,
) -> PMMHResult:
    """PMMH on (mu, phi, sigma_eta) for the AR(1)-SV model.

    Proposals: random walk on (mu, atanh(phi), log sigma_eta).
    The acceptance ratio uses the *particle-estimated* marginal loglik —
    the pseudo-marginal scheme. Deterministic given ``seed``.
    """
    r = _check_returns(returns)
    if n_iter < 20:
        raise ValueError("n_iter >= 20 required")
    if n_particles < 10:
        raise ValueError("n_particles >= 10 required")
    if not 0 < burnin_frac < 0.9:
        raise ValueError("burnin_frac in (0, 0.9)")
    rng = np.random.default_rng(seed)

    # reparameterize: z = (mu, atanh(phi), log sigma_eta)
    def to_z(theta: FloatArray) -> FloatArray:
        return np.array([theta[0], np.arctanh(theta[1]), log(theta[2])])

    def from_z(z: FloatArray) -> FloatArray:
        return np.array([z[0], np.tanh(z[1]), exp(z[2])])

    th0 = np.array([-4.0, 0.9, 0.2]) if theta0 is None else np.asarray(theta0, dtype=float).ravel()
    if th0.size != 3 or not -0.999 < th0[1] < 0.999 or th0[2] <= 0:
        raise ValueError("theta0 must be (mu, phi in (-1,1), sigma_eta>0)")
    psd = (
        np.array([0.15, 0.2, 0.15])
        if proposal_sd is None
        else np.asarray(proposal_sd, dtype=float).ravel()
    )
    if psd.size != 3 or (psd <= 0).any():
        raise ValueError("proposal_sd must be 3 positive scales")

    z = to_z(th0)
    seeds = rng.integers(0, 2**31 - 1, size=n_iter * 2 + 2)
    th = from_z(z)
    ll_cur = sv_pf_loglik(
        r,
        float(th[0]),
        float(th[1]),
        float(th[2]),
        n_particles=n_particles,
        seed=int(seeds[0]),
    )
    chain = np.empty((n_iter, 3))
    chain_ll = np.empty(n_iter)
    accepted = 0
    for i in range(n_iter):
        z_prop = z + psd * rng.standard_normal(3)
        th_prop = from_z(z_prop)
        if abs(th_prop[1]) >= 0.999 or th_prop[2] <= 0 or not np.isfinite(th_prop).all():
            chain[i], chain_ll[i] = from_z(z), ll_cur
            continue
        ll_prop = sv_pf_loglik(
            r,
            float(th_prop[0]),
            float(th_prop[1]),
            float(th_prop[2]),
            n_particles=n_particles,
            seed=int(seeds[2 * i + 1]),
        )
        if ll_prop >= ll_cur or rng.random() < exp(min(ll_prop - ll_cur, 0.0)):
            z, ll_cur = z_prop, ll_prop
            accepted += 1
        chain[i], chain_ll[i] = from_z(z), ll_cur
    burn = int(n_iter * burnin_frac)
    post = chain[burn:]
    return PMMHResult(
        post_mean=post.mean(axis=0),
        post_std=post.std(axis=0),
        acceptance_rate=accepted / n_iter,
        n_iter=n_iter,
        chain_loglik=chain_ll,
        effective_sizes=np.array([_ess_1d(post[:, j]) for j in range(3)]),
    )


def bench_pmcmc_sv(
    seed: int = 20260131,
    *,
    n_obs: int = 600,
    pf_particles: int = 100,
    pf_particles_hi: int = 800,
    n_iter: int = 250,
    n_particles: int = 200,
) -> dict[str, float]:
    """SYNTHETIC bench for the PMCMC-SV machinery. Correctness only."""
    out: dict[str, float] = {}
    true_mu, true_phi, true_eta = -4.0, 0.95, 0.15
    r, _ = simulate_sv(n_obs, mu=true_mu, phi=true_phi, sigma_eta=true_eta, seed=seed)
    # PF loglik stability vs particle count
    l1 = sv_pf_loglik(r, true_mu, true_phi, true_eta, n_particles=pf_particles, seed=seed)
    l2 = sv_pf_loglik(r, true_mu, true_phi, true_eta, n_particles=pf_particles_hi, seed=seed)
    out["synthetic_loglik_stability"] = abs(l1 - l2)
    # PMMH posterior recovery (short chain — smoke-scale)
    res = pmmh_sv(r, n_iter=n_iter, n_particles=n_particles, seed=seed)
    out["synthetic_post_mu_err"] = abs(float(res.post_mean[0]) - true_mu)
    out["synthetic_post_phi_err"] = abs(float(res.post_mean[1]) - true_phi)
    out["synthetic_post_sigma_err"] = abs(float(res.post_mean[2]) - true_eta)
    out["synthetic_acceptance_rate"] = res.acceptance_rate
    out["synthetic_ess_min"] = float(res.effective_sizes.min())
    # determinism. Defaults reproduce the historical 200/60/80 pair.
    det_obs = min(200, n_obs)
    det_iter = min(60, n_iter)
    det_particles = min(80, n_particles)
    a = pmmh_sv(r[:det_obs], n_iter=det_iter, n_particles=det_particles, seed=seed)
    b = pmmh_sv(r[:det_obs], n_iter=det_iter, n_particles=det_particles, seed=seed)
    out["synthetic_determinism"] = float(
        np.allclose(a.post_mean, b.post_mean) and a.acceptance_rate == b.acceptance_rate
    )
    return out
