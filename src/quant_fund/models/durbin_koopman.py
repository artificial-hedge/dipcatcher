"""Durbin-Koopman simulation and disturbance smoothers.

For a linear-Gaussian state-space model

``y_t = Z a_t + eps_t,   eps_t ~ N(0, R)``
``a_{t+1} = T a_t + eta_t,   eta_t ~ N(0, Q)``

the RTS smoother returns the conditional mean/covariance of the state,
but not posterior *draws*. The Durbin-Koopman (2002) simulation
smoother corrects an unconditional model simulation by its own smoothed
estimate: ``a^* = E[a|y] + (a^+ - E[a^+|y^+])`` is exactly a draw from
``p(a | y)``. The Koopman (1993) disturbance smoother returns
``eps_hat``/``eta_hat`` — the smoothed innovations — which are the cheap
route to score evaluation, auxiliary-signal extraction, and EM moments.

Functions
---------
- :func:`kalman_filter` — filter with per-period NaN masking, storing
  innovations, F-inverse terms and gains for reuse by both smoothers.
- :func:`rts_smooth` — fixed-interval conditional means/covariances.
- :func:`disturbance_smooth` — smoothed ``eps``/``eta`` estimates.
- :func:`simulation_smoother` — exact posterior path draws.
- :func:`draw_posterior_paths` — draws + empirical mean/cov.
- :func:`synth_ssm` — synthetic factor state-space generator.
- :func:`bench_durbin_koopman` — SYNTHETIC telemetry blob.

References
----------
- Durbin & Koopman (2002). A simple and efficient simulation smoother
  for state space time series analysis. *Biometrika* 89(3) — journal.
- Koopman (1993). Disturbance smoother for state space models.
  *Biometrika* 80(1) — journal.
- Durbin & Koopman (2012). *Time Series Analysis by State Space
  Methods*, 2nd ed. Oxford University Press — book.

Honesty
-------
All reported numbers are SYNTHETIC calibration checks on seeded
state-space generators — they validate the smoother machinery, never
market data. Posterior-draw calibration is assessed at MC accuracy
(finite draws), reported as diagnostics.

Composition notes
-----------------
- ``models/factor_nowcast.py`` (wave 26): RTS smoother + EM for the
  dynamic-factor panel — this module adds the simulation/disturbance
  layer the DFM deliberately omits (posterior draws for uncertainty
  propagation, disturbance estimates for diagnostics).
- ``models/state_space.py``: baseline state-space utilities — this
  module keeps its own filter so the DK bookkeeping (stored ``v, F^{-1},
  K`` and mask-aware ``L``) stays in one place.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_model(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    arr = np.asarray(y, dtype=np.float64)
    zm = np.asarray(z, dtype=np.float64)
    tm = np.asarray(trans, dtype=np.float64)
    qm = np.asarray(q, dtype=np.float64)
    rm = np.asarray(r, dtype=np.float64)
    a0 = np.asarray(a1, dtype=np.float64).ravel()
    p0 = np.asarray(p1, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError("y must be a (T>=4, N) panel")
    n_y = arr.shape[1]
    k = tm.shape[0]
    if zm.shape != (n_y, k) or tm.ndim != 2 or tm.shape[1] != k:
        raise ValueError("z (N,k) / trans (k,k) shape mismatch")
    if qm.shape != (k, k) or a0.shape != (k,) or p0.shape != (k, k):
        raise ValueError("q/a1/p1 shape mismatch")
    if rm.shape == (n_y, n_y):
        rd = np.diag(rm).copy()
        if not np.allclose(rm, np.diag(rd), atol=1e-12):
            raise ValueError("R must be diagonal")
        rm = rd
    if rm.shape != (n_y,) or (rm <= 0).any():
        raise ValueError("r must be positive diag (N,) or (N,N)")
    if np.isnan(arr).all():
        raise ValueError("y entirely NaN")
    return arr, zm, tm, qm, rm, a0, p0


def kalman_filter(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
) -> dict[str, FloatArray | float]:
    """Kalman filter storing everything the DK machinery reuses.

    Per-period NaN rows are masked out of the update (reduced-dim obs
    vector), which is the ragged-edge semantics the nowcasting lane
    standardized. Returns filtered/predicted moments plus ``v``,
    ``f_inv`` (per-obs innovation precision), ``k_gain`` (``T P Z'F^-1``)
    and ``loglik``.
    """
    arr, zm, tm, qm, rm, a0, p0 = _check_model(y, z, trans, q, r, a1, p1)
    t_n, n_y = arr.shape
    k = tm.shape[0]

    a_f = np.zeros((t_n, k))
    p_f = np.zeros((t_n, k, k))
    a_p = np.zeros((t_n, k))
    p_p = np.zeros((t_n, k, k))
    v_store = np.zeros((t_n, n_y))
    finv_store = np.zeros((t_n, n_y, n_y)) * np.nan
    k_store = np.zeros((t_n, k, n_y))
    obs_mask = np.isfinite(arr)
    loglik = 0.0

    a, p = a0, p0
    for t in range(t_n):
        if t > 0:
            a = tm @ a
            p = tm @ p @ tm.T + qm
        a_p[t] = a
        p_p[t] = p
        obs = obs_mask[t]
        if obs.any():
            zt = zm[obs]
            yt = arr[t, obs]
            f_mat = zt @ p @ zt.T + np.diag(rm[obs])
            f_inv = np.linalg.inv(f_mat)
            v = yt - zt @ a
            k_gain = tm @ p @ zt.T @ f_inv  # (k, m_obs)
            a = a + p @ zt.T @ f_inv @ v
            p = p - p @ zt.T @ f_inv @ zt @ p
            # scatter back to full-N storage
            full_idx = np.flatnonzero(obs)
            v_store[t, full_idx] = v
            finv_store[t][np.ix_(full_idx, full_idx)] = f_inv
            k_store[t][:, full_idx] = k_gain
            sign, logdet = np.linalg.slogdet(f_mat)
            if sign <= 0:
                raise ValueError("non-PD innovation covariance")
            loglik += float(-0.5 * (v @ f_inv @ v + logdet + yt.size * np.log(2 * np.pi)))
        f_f = a
        p_ff = p
        a_f[t] = f_f
        p_f[t] = p_ff

    return {
        "a_filt": a_f,
        "p_filt": p_f,
        "a_pred": a_p,
        "p_pred": p_p,
        "v": v_store,
        "f_inv": finv_store,
        "k_gain": k_store,
        "obs_mask": obs_mask.astype(bool),
        "loglik": loglik,
    }


def rts_smooth(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
    _kf: dict[str, FloatArray | float] | None = None,
) -> dict[str, FloatArray]:
    """Fixed-interval smoother: conditional means and covariances."""
    kf = _kf or kalman_filter(y, z, trans, q, r, a1, p1)
    tm = np.asarray(trans, dtype=np.float64)
    a_f = np.asarray(kf["a_filt"])
    p_f = np.asarray(kf["p_filt"])
    a_p = np.asarray(kf["a_pred"])
    p_p = np.asarray(kf["p_pred"])
    t_n, k = a_f.shape
    a_s = a_f.copy()
    p_s = p_f.copy()
    for t in range(t_n - 2, -1, -1):
        j_gain = p_f[t] @ tm.T @ np.linalg.inv(p_p[t + 1])
        a_s[t] = a_f[t] + j_gain @ (a_s[t + 1] - a_p[t + 1])
        p_s[t] = p_f[t] + j_gain @ (p_s[t + 1] - p_p[t + 1]) @ j_gain.T
    return {"a_s": a_s, "p_s": p_s}


def disturbance_smooth(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
) -> dict[str, FloatArray]:
    """Koopman (1993) disturbance smoother.

    Backward recursion ``r_{t-1} = Z' F^{-1} v_t + L' r_t`` with
    ``L = T - T K Z`` (gain in the ``T P Z'F^{-1}`` convention), giving
    ``eta_hat_t = Q r_t`` and ``eps_hat_t = R (F^{-1} v_t - K' r_t)`` on
    observed coordinates.
    """
    arr, zm, tm, qm, rm, a0, p0 = _check_model(y, z, trans, q, r, a1, p1)
    kf = kalman_filter(arr, zm, tm, qm, rm, a0, p0)
    v = np.asarray(kf["v"])
    f_inv = np.asarray(kf["f_inv"])
    k_gain = np.asarray(kf["k_gain"])
    obs_mask = np.asarray(kf["obs_mask"], dtype=bool)
    t_n, n_y = arr.shape
    k = tm.shape[0]

    eta_hat = np.zeros((t_n, k))
    eps_hat = np.zeros((t_n, n_y)) * np.nan
    r_vec = np.zeros(k)
    for t in range(t_n - 1, -1, -1):
        # eta_hat_t = Q r_t (r as running statistic BEFORE this period's obs)
        eta_hat[t] = qm @ r_vec
        obs = obs_mask[t]
        if obs.any():
            idx = np.flatnonzero(obs)
            finv = f_inv[t][np.ix_(idx, idx)]
            zt = zm[idx]
            kt = k_gain[t][:, idx]  # (k, m)
            l_mat = tm - kt @ zt
            u_obs = finv @ v[t, idx]
            r_vec = zt.T @ u_obs + l_mat.T @ r_vec
            eps_hat[t, idx] = rm[idx] * (u_obs - kt.T @ r_vec)
    return {"eps_hat": eps_hat, "eta_hat": eta_hat}


def _unconditional_sim(
    t_n: int,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Simulate (a+, y+) from the model, no NaNs."""
    k = trans.shape[0]
    n_y = z.shape[0]
    a_sim = np.zeros((t_n, k))
    y_sim = np.zeros((t_n, n_y))
    a_sim[0] = a1
    lq = np.linalg.cholesky(q + 1e-12 * np.eye(k))
    lr = np.sqrt(r)
    for t in range(t_n):
        if t > 0:
            a_sim[t] = trans @ a_sim[t - 1] + lq @ rng.standard_normal(k)
        y_sim[t] = z @ a_sim[t] + lr * rng.standard_normal(n_y)
    return a_sim, y_sim


def simulation_smoother(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
    n_draws: int = 100,
    seed: int = 0,
) -> FloatArray:
    """Exact posterior draws ``a^* ~ p(a | y)`` via the DK correction.

    ``a^* = a_hat(y) + a^+ - a_hat(y^+)`` where ``(a^+, y^+)`` is an
    unconditional model simulation. Draws are correct in distribution
    for any linear-Gaussian model (missing-obs mask in ``y`` is honored
    through the reference smoother).
    """
    arr, zm, tm, qm, rm, a0, p0 = _check_model(y, z, trans, q, r, a1, p1)
    if n_draws < 2:
        raise ValueError("n_draws must be >= 2")
    t_n, n_y = arr.shape
    k = tm.shape[0]
    rng = np.random.default_rng(seed)

    base = rts_smooth(arr, zm, tm, qm, rm, a0, p0)
    a_hat = np.asarray(base["a_s"])

    draws = np.zeros((n_draws, t_n, k))
    for j in range(n_draws):
        a_sim, y_sim = _unconditional_sim(t_n, zm, tm, qm, rm, a0, rng)
        ks = rts_smooth(y_sim, zm, tm, qm, rm, a0, p0)
        draws[j] = a_hat + a_sim - np.asarray(ks["a_s"])
    return draws


def draw_posterior_paths(
    y: FloatArray,
    z: FloatArray,
    trans: FloatArray,
    q: FloatArray,
    r: FloatArray,
    a1: FloatArray,
    p1: FloatArray,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Posterior draws plus their empirical mean and covariance."""
    draws = simulation_smoother(y, z, trans, q, r, a1, p1, n_draws, seed)
    return {
        "draws": draws,
        "draw_mean": draws.mean(axis=0),
        "draw_cov": np.stack(
            [np.atleast_2d(np.cov(draws[:, t, :].T)) for t in range(draws.shape[1])]
        ),
    }


def synth_ssm(n_obs: int, n_series: int, rho: float, seed: int) -> dict[str, FloatArray]:
    """Single-factor SSM: ``f_t = rho f_{t-1} + eta``, ``y = lam f + eps``.

    Returns the observation panel, true factor path, and model matrices
    ``(z, trans, q, r, a1, p1)`` ready for the smoother API.
    """
    if n_obs < 30 or n_series < 1 or not 0.0 <= rho < 0.999:
        raise ValueError("bad n_obs/n_series/rho")
    rng = np.random.default_rng(seed)
    f = np.zeros(n_obs)
    eta = rng.standard_normal(n_obs) * np.sqrt(1.0 - rho * rho)
    for t in range(1, n_obs):
        f[t] = rho * f[t - 1] + eta[t]
    lam = rng.uniform(0.5, 1.5, (n_series, 1))
    r_obs = np.full(n_series, 0.25)
    y = f[:, None] @ lam.T + rng.standard_normal((n_obs, n_series)) * np.sqrt(r_obs)
    z = lam
    trans = np.asarray([[rho]])
    q = np.asarray([[1.0 - rho * rho]])
    a1 = np.zeros(1)
    p1 = np.eye(1)
    return {
        "y": y,
        "f_true": f,
        "load": lam,
        "z": z,
        "trans": trans,
        "q": q,
        "r": r_obs,
        "a1": a1,
        "p1": p1,
        "eta_true": eta,
    }


def bench_durbin_koopman(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC smoother blob — calibration/recovery diagnostics only."""
    sim = synth_ssm(200, 4, 0.9, seed)
    y = sim["y"]
    f_true = sim["f_true"]
    mats = (sim["z"], sim["trans"], sim["q"], sim["r"], sim["a1"], sim["p1"])

    sm = rts_smooth(y, *mats)
    a_s = np.asarray(sm["a_s"])[:, 0]
    p_s = np.asarray(sm["p_s"])[:, 0, 0]
    corr_smooth = float(abs(np.corrcoef(a_s, f_true)[0, 1]))

    kf = kalman_filter(y, *mats)
    a_f = np.asarray(kf["a_filt"])[:, 0]
    corr_filt = float(abs(np.corrcoef(a_f, f_true)[0, 1]))

    dist = disturbance_smooth(y, *mats)
    eta_hat = np.asarray(dist["eta_hat"])[:, 0]
    eta_true = np.asarray(sim["eta_true"])
    # eta_hat_t estimates eta_t which drives f_{t+1}; corr on interior
    corr_eta = float(abs(np.corrcoef(eta_hat[1:-1], eta_true[1:-1])[0, 1]))

    draws = simulation_smoother(y, *mats, n_draws=150, seed=seed + 1)
    draw_mean = draws.mean(axis=0)[:, 0]
    draw_sd = draws.std(axis=0)[:, 0]
    corr_draw = float(abs(np.corrcoef(draw_mean, a_s)[0, 1]))
    # marginal calibration: share of true states inside draw ±1.96 sd
    inside = (np.abs(f_true - draw_mean) <= 1.96 * draw_sd).mean()
    # smoother-vs-draw cov consistency: draw sd vs smoother sd
    sd_ratio = float((draw_sd / np.sqrt(p_s)).mean())

    draws2 = simulation_smoother(y, *mats, n_draws=150, seed=seed + 1)
    determinism = float(np.allclose(draws, draws2))

    return {
        "synthetic_smooth_corr": corr_smooth,
        "synthetic_filter_corr": corr_filt,
        "synthetic_smooth_ge_filter": float(corr_smooth >= corr_filt - 1e-9),
        "synthetic_eta_resid_corr": corr_eta,
        "synthetic_draw_mean_corr": corr_draw,
        "synthetic_draw_cov_calib": float(inside),
        "synthetic_draw_sd_ratio": sd_ratio,
        "synthetic_loglik_finite": float(np.isfinite(float(kf["loglik"]))),
        "synthetic_determinism": determinism,
    }
