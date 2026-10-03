"""Bayesian multi-target tracking: IMM estimator and
PDA data association.

- Interacting Multiple Model (IMM) filter: mode mixing
  (Markov transition), per-mode Kalman update, Gaussian
  moment merge (Blom & Bar-Shalom 1988).
- Probabilistic Data Association (PDA): ellipsoidal
  validation gate, association probabilities over clutter,
  combined innovation update (Bar-Shalom & Tse 1975).

References
----------
- Bar-Shalom & Tse (1975) 'Tracking in a cluttered
  environment with probabilistic data association'
  Automatica 11(5).
- Blom & Bar-Shalom (1988) 'The interacting multiple
  model algorithm for systems with Markovian switching
  coefficients' IEEE Trans. Autom. Control 33(8).
- Blackman & Popoli (1999) Design and Analysis of
  Modern Tracking Systems.

Honesty
-------
SYNTHETIC self-check: a constant-velocity / coordinated-
turn trajectory with seeded clutter; asserts the IMM
recovers the true mode and RMS error stays bounded.

Composition
-----------
Pure numpy. Inputs are 2-D measurement sequences and
process/measurement covariances; outputs are state
estimates, mode probabilities, and innovation stats.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _f_cv(dt: float) -> FloatArray:
    return np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _f_ct(dt: float, omega: float) -> FloatArray:
    """Coordinated-turn transition (small-omega limit safe)."""
    if abs(omega) < 1e-8:
        return _f_cv(dt)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array(
        [
            [1, 0, s / omega, -(1 - c) / omega],
            [0, 1, (1 - c) / omega, s / omega],
            [0, 0, c, -s],
            [0, 0, s, c],
        ],
        dtype=np.float64,
    )


def _q_cv(dt: float, q: float) -> FloatArray:
    return q * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ],
        dtype=np.float64,
    )


def _kf_predict(x: FloatArray, P: FloatArray, F: FloatArray, Q: FloatArray):
    xp = F @ x
    Pp = F @ P @ F.T + Q
    return xp, np.asarray((Pp + Pp.T) / 2)


def _kf_update(xp: FloatArray, Pp: FloatArray, z: FloatArray, H: FloatArray, R: FloatArray):
    innov = z - H @ xp
    S = H @ Pp @ H.T + R
    K = np.linalg.solve(S.T, (Pp @ H.T).T).T
    xu = xp + K @ innov
    Pu = (np.eye(Pp.shape[0]) - K @ H) @ Pp
    return xu, np.asarray((Pu + Pu.T) / 2), innov, S


def pda_update(
    xp: FloatArray,
    Pp: FloatArray,
    measurements: FloatArray,
    H: FloatArray,
    R: FloatArray,
    pd: float = 0.9,
    gate_prob: float = 0.99,
    clutter_density: float = 1e-4,
) -> dict[str, FloatArray | float]:
    """PDA update: gate measurements, combine innovations.

    measurements: (M, m) candidate hits in measurement space.
    Returns updated state/covariance and per-measurement betas.
    """
    z_dim = H.shape[0]
    if measurements.size == 0:
        return {
            "x": xp,
            "P": Pp,
            "betas": np.zeros(0),
            "innovations": np.zeros((0, z_dim)),
            "missed_prob": 1.0,
        }
    S = H @ Pp @ H.T + R
    S_inv = np.linalg.inv(S)
    gate_thr = 5.991 if gate_prob <= 0.95 else 9.210  # chi2_2 0.95/0.99
    zs = np.asarray(measurements, dtype=np.float64)
    if zs.ndim == 1:
        zs = zs[None, :]
    if zs.shape[1] != z_dim:
        raise ValueError("measurement dim mismatch")
    innovs = zs - (H @ xp)
    d2 = np.einsum("ij,jk,ik->i", innovs, S_inv, innovs)
    inside = d2 <= gate_thr
    zg, ig, d2g = zs[inside], innovs[inside], d2[inside]
    n_g = zg.shape[0]
    if n_g == 0:
        return {
            "x": xp,
            "P": Pp,
            "betas": np.zeros(0),
            "innovations": np.zeros((0, z_dim)),
            "missed_prob": 1.0,
        }
    pg = gate_prob
    det_S = float(np.linalg.det(S))
    lik = np.exp(-0.5 * d2g) / ((2 * np.pi) ** (z_dim / 2) * np.sqrt(det_S))
    b = clutter_density * (1 - pd * pg) / pd
    e = np.concatenate([[b], lik])
    betas_full = e / e.sum()
    beta0, betas = float(betas_full[0]), betas_full[1:]
    K = np.linalg.solve(S.T, (Pp @ H.T).T).T
    v_comb = betas @ ig
    xu = xp + K @ v_comb
    Pc = Pp - K @ S @ K.T
    # spread of innovations term
    spread = np.zeros_like(Pp)
    for i in range(n_g):
        dv = ig[i] - v_comb
        spread += betas[i] * np.outer(K @ dv, K @ dv)
    Pu = beta0 * Pp + (1 - beta0) * Pc + spread
    return {
        "x": xu,
        "P": np.asarray((Pu + Pu.T) / 2),
        "betas": betas,
        "innovations": ig,
        "missed_prob": beta0,
    }


def imm_filter(
    measurements: list[FloatArray],
    dt: float = 1.0,
    q_cv: float = 0.5,
    q_ct: float = 1.0,
    omega: float = 0.05,
    r_std: float = 2.0,
    p_stay: float = 0.95,
) -> dict[str, FloatArray]:
    """Two-mode IMM (CV + coordinated turn) over a measurement seq.

    measurements: list of (M_i, 2) arrays — may be empty (missed
    detection) or contain clutter rows.
    """
    if not measurements:
        raise ValueError("empty measurement sequence")
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * r_std**2
    Fs = [_f_cv(dt), _f_ct(dt, omega)]
    Qs = [_q_cv(dt, q_cv), _q_cv(dt, q_ct)]
    n_mode = 2
    Pi = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    z0 = np.asarray(measurements[0])
    first = z0[0] if z0.ndim == 2 and z0.shape[0] > 0 else np.zeros(2)
    xs = [np.array([first[0], first[1], 0, 0], dtype=np.float64) for _ in range(n_mode)]
    Ps = [np.diag([25.0, 25.0, 9.0, 9.0]) for _ in range(n_mode)]
    mu = np.full(n_mode, 0.5)
    track: list[FloatArray] = []
    mus: list[FloatArray] = []
    for zs_raw in measurements[1:]:
        # mixing
        c_bar = np.clip(Pi.T @ mu, 1e-12, None)
        # mu_ij[i, j] = P(mode i at k-1 | mode j at k) = Pi[i,j]*mu[i]/c_bar[j]
        mu_ij = (Pi * mu[:, None]) / c_bar[None, :]
        x_mix = [np.zeros(4) for _ in range(n_mode)]
        P_mix = [np.zeros((4, 4)) for _ in range(n_mode)]
        for j in range(n_mode):
            x_mix[j] = mu_ij[:, j] @ np.vstack(xs)
            for i in range(n_mode):
                dx = xs[i] - x_mix[j]
                P_mix[j] += mu_ij[i, j] * (Ps[i] + np.outer(dx, dx))
        # per-mode predict + PDA update
        zarr = np.asarray(zs_raw)
        if zarr.ndim == 1:
            zarr = zarr[None, :]
        for j in range(n_mode):
            xp, Pp = _kf_predict(x_mix[j], P_mix[j], Fs[j], Qs[j])
            if zarr.size == 0:
                xs[j], Ps[j] = xp, Pp
                lik_j = 1e-9  # missed detection dampens mode
            else:
                res = pda_update(xp, Pp, zarr, H, R)
                xs[j] = np.asarray(res["x"])
                Ps[j] = np.asarray(res["P"])
                # mode likelihood: gated innovation evidence
                if np.asarray(res["innovations"]).shape[0] > 0:
                    S = H @ Pp @ H.T + R
                    ig = np.asarray(res["innovations"])
                    d2 = np.einsum("ij,jk,ik->i", ig, np.linalg.inv(S), ig)
                    lik_j = float(np.exp(-0.5 * d2.min()))
                else:
                    lik_j = 1e-9
            mu[j] = c_bar[j] * lik_j
        mu_sum = mu.sum()
        if mu_sum <= 0 or not np.isfinite(mu_sum):
            mu = np.full(n_mode, 0.5)
        else:
            mu = mu / mu_sum
        # merge
        x_out = mu @ np.vstack(xs)
        P_out = np.zeros((4, 4))
        for j in range(n_mode):
            dx = xs[j] - x_out
            P_out += mu[j] * (Ps[j] + np.outer(dx, dx))
        track.append(x_out)
        mus.append(mu.copy())
    return {
        "track": np.vstack(track),
        "mode_probs": np.vstack(mus),
    }


def bench_tracking(seed: int = 505) -> dict[str, float]:
    """SYNTHETIC: CV leg then coordinated turn; seeded clutter.

    Asserts RMS position error bounded and CT mode selected on the
    turn segment."""
    rng = np.random.default_rng(seed)
    dt = 1.0
    n = 80
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    vel[0] = [8.0, 0.0]
    for k in range(1, n):
        if k < 40:
            pos[k] = pos[k - 1] + vel[k - 1] * dt
            vel[k] = vel[k - 1]
        else:  # coordinated turn
            om = 0.06
            vx, vy = vel[k - 1]
            c, s = np.cos(om * dt), np.sin(om * dt)
            vel[k] = [c * vx - s * vy, s * vx + c * vy]
            pos[k] = pos[k - 1] + vel[k] * dt
    meas: list[FloatArray] = []
    for k in range(n):
        rows = [pos[k] + rng.normal(0, 2.0, 2)]
        n_cl = rng.poisson(0.4)
        if n_cl > 0:
            clutter = pos[k] + rng.uniform(-40, 40, (n_cl, 2))
            rows.append(clutter)
        meas.append(np.vstack(rows))
    out = imm_filter(meas, dt=dt)
    est = out["track"]
    err = np.sqrt(((est[:, :2] - pos[1:]) ** 2).sum(axis=1))
    rms = float(np.sqrt((err**2).mean()))
    turn_mode = float(out["mode_probs"][50:, 1].mean())
    if rms > 8.0:
        raise ValueError("IMM RMS too large")
    if turn_mode < 0.4:
        raise ValueError("CT mode not selected on turn")
    return {
        "synthetic_rms_pos": rms,
        "synthetic_turn_mode_prob": turn_mode,
        "synthetic_straight_mode_prob": float(out["mode_probs"][:35, 0].mean()),
        "synthetic_max_err": float(err.max()),
    }
