"""Strapdown INS mechanization — quaternion attitude, NED velocity/ (SYNTHETIC)
position integration. Single-interval rotation-vector attitude update
(sculling/coning terms omitted); Coriolis and transport-rate terms
included for short horizons via the Earth-rate approximation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_G = 9.80665
_OMEGA_E = 7.292115e-5  # rad/s


def quat_from_rotvec(rv: FloatArray) -> FloatArray:
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        q = np.concatenate([[1.0], rv / 2.0])
        return np.asarray(q, dtype=np.float64)
    ax = rv / th
    return np.asarray(np.concatenate([[np.cos(th / 2)], ax * np.sin(th / 2)]), dtype=np.float64)


def quat_mul(q: FloatArray, r: FloatArray) -> FloatArray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def dcm_from_quat(q: FloatArray) -> FloatArray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mech_step(
    q: FloatArray,
    v: FloatArray,
    p: FloatArray,
    dtheta: FloatArray,
    dvel: FloatArray,
    lat: float,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One strapdown epoch.

    dtheta/dvel: IMU increments (body frame). p = [lat, lon, h] (deg,
    deg, m). Returns updated (q, v_ned, p).
    """
    latr = np.deg2rad(lat)
    C_bn = dcm_from_quat(q).T  # body → nav
    # earth + transport rate as nav-frame rotation compensation
    w_ie = np.array([_OMEGA_E * np.cos(latr), 0.0, -_OMEGA_E * np.sin(latr)])
    rn = 6378137.0 + p[2]
    w_en = np.array([v[1] / rn, -v[0] / rn, -v[1] * np.tan(latr) / rn])
    q_nav = quat_from_rotvec(-(w_ie + w_en) * dt)
    q = quat_mul(q_nav, quat_mul(q, quat_from_rotvec(dtheta)))
    q = q / np.linalg.norm(q)
    g = np.array([0.0, 0.0, _G])
    dv_nav = C_bn @ dvel + (g - np.cross(2 * w_ie + w_en, v)) * dt
    v_new = v + dv_nav
    rn_h = rn
    dp = np.array([v[0] / (rn_h), v[1] / (rn_h * np.cos(latr)), -v[2]]) * dt
    p_new = p + np.rad2deg(np.array([dp[0], dp[1], 0.0])) + np.array([0, 0, dp[2]])
    return q, v_new, p_new


def bench_strapdown(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: stationary IMU holds position/attitude; pure yaw rate
    recovers heading; constant accel integrates consistently."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    dt = 0.01
    q = np.array([1.0, 0, 0, 0])
    v = np.zeros(3)
    p = np.array([45.0, -75.0, 100.0])
    # stationary: dvel compensates gravity (accel measures +g up in body)
    dvel_hold = np.array([0.0, 0.0, -_G]) * dt  # specific force upward
    for _ in range(1000):
        q, v, p = mech_step(
            q,
            v,
            p,
            np.zeros(3) + rng.normal(0, 1e-6, 3),
            dvel_hold + rng.normal(0, 1e-5, 3),
            p[0],
            dt,
        )
    out["synthetic_strapdown_drift_m"] = float(np.abs(v).max())
    out["synthetic_strapdown_pos_err_deg"] = float(np.linalg.norm(p[:2] - np.array([45.0, -75.0])))
    # yaw test: 1 rad/s z-body for 1 s → heading ≈ 1 rad
    q = np.array([1.0, 0, 0, 0])
    v = np.zeros(3)
    p = np.array([0.0, 0.0, 0.0])
    for _ in range(100):
        q, v, p = mech_step(
            q, v, p, np.array([0, 0, 1.0]) * dt, np.array([0, 0, -_G]) * dt, p[0], dt
        )
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    out["synthetic_strapdown_yaw_err"] = abs(abs(yaw) - 1.0)
    out["synthetic_strapdown_quat_norm_err"] = float(abs(np.linalg.norm(q) - 1))
    return out


if __name__ == "__main__":
    print(bench_strapdown())
