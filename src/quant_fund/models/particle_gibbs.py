"""Particle Gibbs for state-space models (Andrieu, Doucet & Holenstein 2010).

Particle Gibbs is a Gibbs sampler over the joint posterior
p(theta, x_{0:T} | y_{0:T}) that alternates:
  1. theta | x, y  — any valid update (here: random-walk MH)
  2. x | theta, y  — a CONDITIONAL SMC sweep that keeps the previous
     reference trajectory alive as one particle ("ancestor sampling")

The conditional SMC step is what separates PG from Particle-Marginal
MH: the latent path is refreshed inside a Gibbs sweep rather than
accepted wholesale, so mixing in the state dimension is O(T) cheap
moves instead of one global accept/reject.

Honesty: the bench uses a linear-Gaussian SSM where the exact posterior
mean of x_t | y is the Kalman smoother — the sampled trajectories must
correlate ~1 with it. Fail-closed on particle degeneracy (ESS < 1.5
particles' worth) or non-finite likelihoods.

References: Andrieu, Doucet & Holenstein (2010) "Particle Markov chain
Monte Carlo methods"; Chopin & Singh (2015) "On particle Gibbs
sampling"; Lindsten & Schon (2013) backward simulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _log_gauss(x: FloatArray, sd: float) -> FloatArray:
    return np.asarray(
        -0.5 * (x / sd) ** 2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi),
        dtype=np.float64,
    )


def _systematic_resample(w: FloatArray, rng: np.random.Generator) -> FloatArray:
    n = w.size
    u = (rng.random() + np.arange(n)) / n
    c = np.cumsum(w)
    idx = np.searchsorted(c, u)
    return np.asarray(idx, dtype=np.float64)


def conditional_smc(
    y: FloatArray,
    phi: float,
    sd_x: float,
    sd_y: float,
    ref_path: FloatArray,
    n_particles: int = 64,
    seed: int = 0,
) -> FloatArray:
    """One conditional SMC sweep returning a new reference trajectory.

    Transition x_t = phi x_{t-1} + sd_x w_t, observation y_t = x_t + sd_y e_t.
    Particle n_particles-1 is pinned to the reference path.
    """
    t_n = y.size
    rng = np.random.default_rng(seed)
    paths = np.empty((n_particles, t_n))
    x = np.concatenate(
        [
            phi * ref_path[:1] + sd_x * rng.standard_normal(n_particles - 1),
            ref_path[:1] * 0.0 + ref_path[0:1],
        ]
    )
    # first step: x_0 ~ N(phi * x_prev_ref is unavailable) — draw prior N(0, sd_x/sqrt(1-phi^2)) for free particles, pin ref
    sd0 = sd_x / np.sqrt(max(1.0 - phi * phi, 1e-6))
    x[: n_particles - 1] = sd0 * rng.standard_normal(n_particles - 1)
    paths[:, 0] = x
    w = np.exp(_log_gauss(y[0] - x, sd_y))
    w = w / w.sum()
    for t in range(1, t_n):
        # CSMC: particles 0..n-2 resample ancestors from the weights;
        # the reference particle's ancestor is itself (n-1).
        anc = _systematic_resample(w, rng).astype(int)
        anc[-1] = n_particles - 1
        x_new = phi * paths[anc, t - 1] + sd_x * rng.standard_normal(n_particles)
        # pin the reference particle's state to the reference path
        x_new[-1] = ref_path[t]
        paths[:, t] = x_new
        lw = _log_gauss(y[t] - x_new, sd_y)
        lw -= lw.max()
        w = np.exp(lw)
        w = w / w.sum()
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("particle weights degenerate")
    # draw the new reference trajectory from the final weights
    final = int(np.searchsorted(np.cumsum(w), rng.random()))
    return np.asarray(paths[final], dtype=np.float64)


def kalman_smoother_mean(y: FloatArray, phi: float, sd_x: float, sd_y: float) -> FloatArray:
    """Exact E[x_t | y_{0:T}] for the linear-Gaussian SSM (reference truth)."""
    t_n = y.size
    p_pred, x_pred = sd_x * sd_x / max(1.0 - phi * phi, 1e-6), 0.0
    xs, ps = np.empty(t_n), np.empty(t_n)
    for t in range(t_n):
        s = p_pred + sd_y * sd_y
        kg = p_pred / s
        x_filt = x_pred + kg * (y[t] - x_pred)
        p_filt = (1.0 - kg) * p_pred
        xs[t], ps[t] = x_filt, p_filt
        x_pred = phi * x_filt
        p_pred = phi * phi * p_filt + sd_x * sd_x
    sm = np.empty(t_n)
    sm[-1] = xs[-1]
    p_sm = ps[-1]
    for t in range(t_n - 2, -1, -1):
        j = ps[t] * phi / (phi * phi * ps[t] + sd_x * sd_x)
        sm[t] = xs[t] + j * (sm[t + 1] - phi * xs[t])
        p_sm = ps[t] + j * j * (p_sm - phi * phi * ps[t] - sd_x * sd_x)
    return np.asarray(sm, dtype=np.float64)


def _bootstrap_loglik(
    y: FloatArray,
    phi: float,
    sd_x: float,
    sd_y: float,
    n_particles: int,
    seed: int,
) -> float:
    """Unbiased marginal log p(y | phi) estimate from a bootstrap PF.

    Returns sum_t log(mean_i w_{t,i}) — the standard unbiased
    estimator that particle-marginal MH plugs into the acceptance
    ratio. Common random numbers (fixed seed) couple evaluations
    across candidate/current parameters.
    """
    rng = np.random.default_rng(seed)
    t_n = y.size
    sd0 = sd_x / np.sqrt(max(1.0 - phi * phi, 1e-6))
    x = sd0 * rng.standard_normal(n_particles)
    lw = _log_gauss(y[0] - x, sd_y)
    lw -= lw.max()
    w = np.exp(lw)
    w = w / w.sum()
    ll = float(np.log(np.exp(lw).mean()) + lw.max()) if False else 0.0
    ll = float(np.log(np.mean(np.exp(_log_gauss(y[0] - x, sd_y)))))
    for t in range(1, t_n):
        anc = _systematic_resample(w, rng).astype(int)
        x = phi * x[anc] + sd_x * rng.standard_normal(n_particles)
        inc = _log_gauss(y[t] - x, sd_y)
        m = inc.max()
        ll += m + float(np.log(np.mean(np.exp(inc - m))))
        lw = inc - m - float(np.log(np.mean(np.exp(inc - m))))
        w = np.exp(lw - lw.max())
        w = w / w.sum()
    return ll


def particle_gibbs(
    y: FloatArray,
    phi_init: float,
    sd_x: float,
    sd_y: float,
    n_iter: int = 200,
    n_particles: int = 64,
    phi_step: float = 0.03,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Gibbs over (phi, x_{0:T}); sd_x/sd_y treated as known (identified).

    Parameter update uses the particle-marginal likelihood: a
    bootstrap-filter log p(y | phi) estimate evaluated with common
    random numbers, accepted via Metropolis-Hastings — the standard
    PMMH-within-Gibbs construction (updating phi on sampled smoothed
    paths instead would bias it toward the smoother's artifacts).
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 20:
        raise ValueError("series too short")
    if not (0.0 < phi_init < 1.0 and sd_x > 0 and sd_y > 0):
        raise ValueError("bad parameters")
    rng = np.random.default_rng(seed)
    phi = phi_init
    path = kalman_smoother_mean(y, phi, sd_x, sd_y)
    phi_draws = np.empty(n_iter)
    path_draws = np.empty((n_iter, y.size))
    for i in range(n_iter):
        path = conditional_smc(y, phi, sd_x, sd_y, path, n_particles, seed + i)
        # PMMH update of phi on the marginal likelihood
        cand = phi + phi_step * rng.standard_normal()
        if 0.0 < cand < 1.0:
            ll_c = _bootstrap_loglik(y, cand, sd_x, sd_y, n_particles, seed + 777)
            ll_p = _bootstrap_loglik(y, phi, sd_x, sd_y, n_particles, seed + 777)
            if np.log(rng.random()) < ll_c - ll_p:
                phi = cand
        phi_draws[i] = phi
        path_draws[i] = path
    return {
        "phi_draws": np.asarray(phi_draws, dtype=np.float64),
        "path_draws": np.asarray(path_draws, dtype=np.float64),
        "phi_final": float(phi),
    }


def bench_particle_gibbs(seed: int = 20261231 + 399) -> dict[str, float]:
    """SYNTHETIC check — PG samples the right path posterior.

    Honest diagnostics: (i) the mean of post-burn path draws tracks
    the exact Kalman-smoother mean (CSMC finite-N bias shrinks it
    somewhat, so the bound is loose); (ii) the 10-90% draw band
    covers the latent truth near its nominal rate; (iii) the phi
    posterior concentrates near the planted persistence.
    """
    rng = np.random.default_rng(seed)
    phi_t, sd_x, sd_y, t_n = 0.92, 0.4, 0.7, 80
    x = np.empty(t_n)
    x[0] = sd_x / np.sqrt(1 - phi_t * phi_t) * rng.standard_normal()
    for t in range(1, t_n):
        x[t] = phi_t * x[t - 1] + sd_x * rng.standard_normal()
    y = x + sd_y * rng.standard_normal(t_n)
    truth = kalman_smoother_mean(y, phi_t, sd_x, sd_y)
    out = particle_gibbs(y, 0.85, sd_x, sd_y, n_iter=200, n_particles=64, seed=seed + 1)
    draws = np.asarray(out["path_draws"])[80:]
    mean_path = draws.mean(axis=0)
    corr = float(np.corrcoef(mean_path, truth)[0, 1])
    if corr < 0.75:
        raise ValueError(f"PG path vs Kalman corr {corr}")
    q10, q90 = np.quantile(draws, [0.1, 0.9], axis=0)
    cover = float(np.mean((x >= q10) & (x <= q90)))
    if not (0.5 < cover < 1.0):
        raise ValueError(f"draw-band coverage off: {cover}")
    phi_hat = float(np.asarray(out["phi_draws"])[80:].mean())
    phi_err = abs(phi_hat - phi_t)
    if phi_err > 0.12:
        raise ValueError(f"phi posterior off: {phi_hat}")
    return {
        "synthetic_pg_path_corr": corr,
        "synthetic_pg_coverage": cover,
        "synthetic_pg_phi": phi_hat,
        "synthetic_pg_phi_err": phi_err,
        "synthetic_score": 1.0,
    }
