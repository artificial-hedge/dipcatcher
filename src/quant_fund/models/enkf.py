"""Sequential ensemble Kalman data assimilation.

Stochastic EnKF (Evensen 2003), deterministic square-root EAKF
(Anderson 2001), and the LETKF ensemble transform (Ott et al. 2004),
plus multiplicative covariance inflation and Gaspari–Cohn localization.
Distinct from ``ensemble_kalman_inversion`` (wave 27): EKI iterates an
ensemble to invert a static parameter-to-observation map; here the
ensemble tracks a *state* forward in time through sequential analysis
steps — filtering, not inversion. The driver ``assimilate_l96`` runs the
standard Lorenz-96 chaotic benchmark.

References
----------
- Lorenz (1996). Predictability — a problem partly solved. *Proc.
  ECMWF Seminar on Predictability*, Shinfield Park, Vol. 1, pp. 1–18.
  (ECMWF proceedings; reprinted in Palmer & Hagedorn 2006, CUP.)
- Evensen (2003). The ensemble Kalman filter: theoretical formulation
  and practical implementation. *Ocean Dynamics* 53:343–367 —
  arXiv:physics/0302019?  (listed as physics/0302019 in some indexes;
  the canonical citation is the journal article).
- Anderson (2001). An ensemble adjustment Kalman filter for data
  assimilation. *Mon. Wea. Rev.* 129:2884–2903.
- Ott, Hunt, Szunyogh, Zimin, Kostelich, Corazza, Kalnay, Patil & Yorke
  (2004). A local ensemble Kalman filter for atmospheric data
  assimilation. *Tellus A* 56:415–428 — arXiv:nlin/0309028.
- Gaspari & Cohn (1999). Construction of correlation functions in two
  and three dimensions. *Q. J. R. Meteorol. Soc.* 125:723–757.

Honesty
-------
The L96 twin experiment is SYNTHETIC: synthetic truth orbit, synthetic
observations. Reported keys are analysis RMSE relative to climatology
and spread/skill ratios — they verify the filter tracks a chaotic
system, never market skill.

Composition notes
-----------------
- ``models/ensemble_kalman_inversion.py``: batch ensemble inversion
  (parameter estimation). This module does sequential state filtering.
- ``models/durbin_koopman.py``: Gaussian linear state-space smoothing;
  EnKF handles *nonlinear* dynamics via the ensemble.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lorenz96(x: FloatArray, f: float = 8.0) -> FloatArray:
    """Lorenz-96 right-hand side: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        raise ValueError("lorenz96 requires >= 3 state components")
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + f
    return dx


def integrate_l96(x0: FloatArray, dt: float, steps: int, f: float = 8.0) -> FloatArray:
    """RK4 integrate Lorenz-96 for ``steps`` steps of size ``dt`` → final state."""
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    if not (dt > 0 and steps > 0):
        raise ValueError("dt and steps must be positive")
    for _ in range(steps):
        k1 = lorenz96(x, f)
        k2 = lorenz96(x + 0.5 * dt * k1, f)
        k3 = lorenz96(x + 0.5 * dt * k2, f)
        k4 = lorenz96(x + dt * k3, f)
        x += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x


def _as_ensemble(e: FloatArray) -> tuple[FloatArray, int, int]:
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] < 2:
        raise ValueError("ensemble must be (n_members >= 2, n_state)")
    if not np.all(np.isfinite(e)):
        raise ValueError("ensemble must be finite")
    return e, e.shape[0], e.shape[1]


def enkf_update(
    ensemble: FloatArray, y: FloatArray, h: FloatArray, r: float, seed: int = 0
) -> FloatArray:
    """Stochastic EnKF analysis: perturbed-observation update.

    ``ensemble`` (K, m) prior members, ``y`` (p,) observation, ``h``
    (K, p) prior ensemble of predicted observations h(x_k), ``r``
    observation-error variance (isotropic). Analysis
    x_k^a = x_k + C_xy (C_yy + R)^{-1} (y + ε_k - h_k), ε_k ~ N(0, R).
    """
    e, k_, m = _as_ensemble(ensemble)
    y = np.asarray(y, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (k_, y.size):
        raise ValueError("h must be (n_members, len(y))")
    if not r > 0:
        raise ValueError("r must be positive")
    mean_x, mean_h = e.mean(axis=0), h.mean(axis=0)
    xa = e - mean_x
    ha = h - mean_h
    c_xy = (xa.T @ ha) / (k_ - 1)
    c_yy = (ha.T @ ha) / (k_ - 1) + r * np.eye(y.size)
    gain = np.linalg.solve(c_yy.T, c_xy.T).T
    rng = np.random.default_rng(seed)
    pert = y + rng.standard_normal((k_, y.size)) * math.sqrt(r)
    return e + (pert - h) @ gain.T


def eakf_update(ensemble: FloatArray, y: FloatArray, h: FloatArray, r: float) -> FloatArray:
    """Deterministic ensemble-adjustment (square-root) update.

    Scalar-observation serial loop (Anderson 2001): for each observed
    component, shift every member's predicted-observation sample so the
    analysis ensemble has the exact posterior mean and variance of the
    scalar Bayesian update, then regress the (obs-space) increment onto
    each state component via the prior joint covariance.
    """
    e, k_, m = _as_ensemble(ensemble)
    y = np.asarray(y, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (k_, y.size):
        raise ValueError("h must be (n_members, len(y))")
    if not r > 0:
        raise ValueError("r must be positive")
    out = e.copy()
    ha = h.copy()
    for j in range(y.size):
        prior = ha[:, j]
        p_var = float(np.var(prior, ddof=1))
        post_var = 1.0 / (1.0 / r + 1.0 / max(p_var, 1e-12))
        post_mean = post_var * (y[j] / r + prior.mean() / max(p_var, 1e-12))
        # deterministic adjust: center at posterior mean, scale anomalies
        adj = (prior - prior.mean()) * math.sqrt(post_var / max(p_var, 1e-12))
        incr_obs = adj + post_mean - prior
        # regress obs-increment onto state space (cov(x, h_j)/var(h_j))
        for i in range(m):
            cov_xh = float(np.cov(out[:, i], prior, ddof=1)[0, 1])
            out[:, i] += cov_xh / max(p_var, 1e-12) * incr_obs
        ha[:, j] = prior + incr_obs
    return out


def letkf_update(
    ensemble: FloatArray, y_local: FloatArray, h_local: FloatArray, r_local: float
) -> FloatArray:
    """LETKF ensemble transform in ensemble space (Ott et al. 2004).

    Pa = ((K-1) I + Yaᵀ R⁻¹ Ya)⁻¹ ; wa = Pa Yaᵀ R⁻¹ (y − h̄);
    analysis anomalies = Xa ( (K-1) Pa )^{1/2}. Returns analysis
    ensemble (K, m).
    """
    e, k_, m = _as_ensemble(ensemble)
    y = np.asarray(y_local, dtype=np.float64).ravel()
    h = np.asarray(h_local, dtype=np.float64)
    if h.shape != (k_, y.size):
        raise ValueError("h_local must be (n_members, len(y_local))")
    if not r_local > 0:
        raise ValueError("r_local must be positive")
    mean_x = e.mean(axis=0)
    xa = e - mean_x
    mean_h = h.mean(axis=0)
    ya = h - mean_h
    pa = np.linalg.inv((k_ - 1) * np.eye(k_) + (ya @ ya.T) / r_local)
    wa = pa @ ya @ ((y - mean_h) / r_local)
    mean_a = mean_x + wa @ xa
    # symmetric sqrt of (K-1) Pa
    evals, evecs = np.linalg.eigh((k_ - 1) * pa)
    evals = np.clip(evals, 0.0, None)
    sqrt_pa = (evecs * np.sqrt(evals)) @ evecs.T
    xa_a = sqrt_pa @ xa
    return mean_a + xa_a


def inflate(ensemble: FloatArray, rho: float = 1.02) -> FloatArray:
    """Multiplicative covariance inflation: anomalies × rho."""
    e, _k, _m = _as_ensemble(ensemble)
    if not rho > 0:
        raise ValueError("rho must be positive")
    return e.mean(axis=0) + rho * (e - e.mean(axis=0))


def localize_gaspari_cohn(dist: FloatArray | float, c: float) -> FloatArray | float:
    """Gaspari-Cohn 5th-order piecewise-polynomial taper.

    1 at dist 0, 0 at dist 2c, smooth in between; zero beyond 2c.
    """
    if not c > 0:
        raise ValueError("c must be positive")
    z = np.asarray(dist, dtype=np.float64)

    def taper(zz: FloatArray) -> FloatArray:
        out = np.zeros_like(zz)
        m1 = zz <= c
        m2 = (zz > c) & (zz <= 2.0 * c)
        u = zz[m1] / c
        out[m1] = -0.25 * u**5 + 0.5 * u**4 + (5.0 / 8.0) * u**3 - (5.0 / 3.0) * u**2 + 1.0
        v = zz[m2] / c
        out[m2] = (
            (1.0 / 12.0) * v**5
            - 0.5 * v**4
            + (5.0 / 8.0) * v**3
            + (5.0 / 3.0) * v**2
            - 5.0 * v
            + 4.0
            - (2.0 / 3.0) / v
        )
        return out

    res = taper(z)
    return float(res) if res.ndim == 0 else res


def assimilate_l96(
    n_state: int = 40,
    n_members: int = 20,
    dt: float = 0.05,
    n_cycles: int = 200,
    obs_freq: int = 1,
    obs_vars: int | None = None,
    r: float = 1.0,
    f: float = 8.0,
    rho: float = 1.04,
    method: str = "enkf",
    spin_steps: int = 400,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Twin experiment: L96 truth + noisy partial obs, filtered by ensemble.

    Returns analysis ``rmse`` (per cycle, state mean RMSE vs truth) and
    ``spread`` (ensemble std vs truth mean), plus ``clim_rmse`` (RMSE of
    the climatological mean) for normalization.
    """
    if method not in ("enkf", "eakf", "letkf"):
        raise ValueError("method must be 'enkf' | 'eakf' | 'letkf'")
    if n_state < 3 or n_members < 2 or n_cycles < 1:
        raise ValueError("invalid assimilation sizes")
    rng = np.random.default_rng(seed)
    # truth: spin up then integrate
    truth = rng.standard_normal(n_state) * 0.1 + f
    truth = integrate_l96(truth, dt, spin_steps, f)
    traj = np.empty((n_cycles + 1, n_state))
    traj[0] = truth
    for c_ in range(n_cycles):
        traj[c_ + 1] = integrate_l96(traj[c_], dt, obs_freq, f)
    clim = traj.mean(axis=0)
    # init ensemble around climatology
    ens = clim + rng.standard_normal((n_members, n_state)) * 1.0
    obs_idx = np.arange(0, n_state, max(1, n_state // (obs_vars or n_state)))[
        : (obs_vars or n_state)
    ]
    rmse = np.empty(n_cycles)
    spread = np.empty(n_cycles)
    obs_every = obs_freq  # obs each cycle boundary
    for c_ in range(n_cycles):
        # forecast each member one cycle (obs_freq L96 steps)
        for k_ in range(n_members):
            ens[k_] = integrate_l96(ens[k_], dt, obs_every, f)
        # observe truth at this cycle's end state
        y = traj[c_ + 1, obs_idx] + rng.standard_normal(obs_idx.size) * math.sqrt(r)
        h = ens[:, obs_idx]
        ens = inflate(ens, rho)
        if method == "enkf":
            ens = enkf_update(ens, y, h, r, seed=seed + 10_000 + c_)
        elif method == "eakf":
            ens = eakf_update(ens, y, h, r)
        else:
            ens = letkf_update(ens, y, h, r)
        rmse[c_] = float(np.sqrt(np.mean((ens.mean(axis=0) - traj[c_ + 1]) ** 2)))
        spread[c_] = float(ens.std(axis=0, ddof=1).mean())
    clim_rmse = float(np.sqrt(np.mean((clim - traj[1:]) ** 2)))
    return {
        "rmse": rmse,
        "spread": spread,
        "clim_rmse": np.array([clim_rmse]),
    }


def bench_enkf(seed: int = 20261231 + 151) -> dict[str, float]:
    """SYNTHETIC data-assimilation telemetry on Lorenz-96."""
    res_e = assimilate_l96(n_cycles=120, n_members=30, rho=1.10, method="enkf", seed=seed)
    res_a = assimilate_l96(n_cycles=120, method="eakf", seed=seed + 1)
    res_l = assimilate_l96(n_cycles=120, method="letkf", n_members=12, seed=seed + 2)
    rmse_e = float(res_e["rmse"].mean())
    rmse_a = float(res_a["rmse"].mean())
    rmse_l = float(res_l["rmse"].mean())
    clim = float(res_e["clim_rmse"][0])
    spread_e = float(res_e["spread"].mean())
    # determinism
    res_b = assimilate_l96(n_cycles=40, method="enkf", seed=seed)
    det = float(
        np.allclose(res_b["rmse"], assimilate_l96(n_cycles=40, method="enkf", seed=seed)["rmse"])
    )
    # localization sanity: GC taper monotone
    zz = np.linspace(0, 4, 9)
    gg = np.asarray(localize_gaspari_cohn(zz, c=1.0))
    mono = float(np.all(np.diff(gg) <= 1e-9))
    return {
        "synthetic_enkf_rmse": rmse_e,
        "synthetic_enkf_rmse_vs_clim": rmse_e / max(clim, 1e-12),
        "synthetic_enkf_spread_skill": spread_e / max(rmse_e, 1e-12),
        "synthetic_eakf_rmse_vs_clim": rmse_a / max(clim, 1e-12),
        "synthetic_letkf_rmse_vs_clim": rmse_l / max(clim, 1e-12),
        "synthetic_bounded": float(np.all(np.isfinite(res_e["rmse"])) and rmse_e < 5.0),
        "synthetic_gc_monotone": mono,
        "synthetic_determinism": det,
    }
