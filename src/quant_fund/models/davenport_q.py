"""Davenport's q-method for Wahba's attitude-determination problem.

SYNTHETIC bench: rotates a set of reference directions by a known DCM,
adds noise, recovers the attitude quaternion, and compares against truth
(angle error) plus orthonormality of the recovered DCM.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_SEED = 20261231 + 925


def davenport_q(b: np.ndarray, r: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Wahba solution via q-method.

    b: (n,3) observed body-frame unit vectors; r: (n,3) reference-frame unit
    vectors; w: optional weights. Returns quaternion (q0,q1,q2,q3), scalar
    first, mapping reference -> body.
    """
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = len(b)
    if w is None:
        w = np.ones(n)
    B = np.zeros((3, 3))
    for i in range(n):
        B += w[i] * np.outer(r[i], b[i])
    S = B + B.T
    sig = float(np.trace(B))
    z = np.asarray([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    K = np.zeros((4, 4))
    K[:3, :3] = S - sig * np.eye(3)
    K[:3, 3] = z
    K[3, :3] = z
    K[3, 3] = sig
    vals, vecs = np.linalg.eigh(K)
    qv = vecs[:, int(np.argmax(vals))]
    q = np.asarray([qv[3], qv[0], qv[1], qv[2]], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.asarray(
        [
            [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
        ],
        dtype=np.float64,
    )


def bench_davenport_q(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # true rotation: 30 deg about axis (1,2,3)/|.|
    ax = np.asarray([1.0, 2.0, 3.0])
    ax /= norm(ax)
    ang = np.deg2rad(30.0)
    q_true = np.concatenate([[np.cos(ang / 2)], ax * np.sin(ang / 2)])
    C_true = quat_to_dcm(q_true)
    r = rng.normal(size=(8, 3))
    r /= norm(r, axis=1, keepdims=True)
    b = (C_true @ r.T).T + rng.normal(scale=1e-4, size=(8, 3))
    b /= norm(b, axis=1, keepdims=True)
    q_est = davenport_q(b, r)
    C_est = quat_to_dcm(q_est)
    # 1) attitude error angle
    err = float(np.arccos(np.clip((np.trace(C_true.T @ C_est) - 1) / 2, -1.0, 1.0)))
    score += 1.0 if err < 1e-3 else 0.0
    # 2) DCM orthonormal
    score += 1.0 if float(norm(C_est.T @ C_est - np.eye(3))) < 1e-10 else 0.0
    # 3) weighted solve improves over unweighted on noisy subset
    b2 = b.copy()
    b2[:4] += rng.normal(scale=0.05, size=(4, 3))
    b2 /= norm(b2, axis=1, keepdims=True)
    q_u = davenport_q(b2, r)
    wts = np.asarray([0.01] * 4 + [1.0] * 4)
    q_w = davenport_q(b2, r, wts)
    e_u = float(np.arccos(np.clip((np.trace(C_true.T @ quat_to_dcm(q_u)) - 1) / 2, -1.0, 1.0)))
    e_w = float(np.arccos(np.clip((np.trace(C_true.T @ quat_to_dcm(q_w)) - 1) / 2, -1.0, 1.0)))
    score += 1.0 if e_w <= e_u + 1e-9 else 0.0
    # 4) degenerate single-vector handled (two-vector solve still exact)
    q2 = davenport_q(b[:2], r[:2])
    e2 = float(np.arccos(np.clip((np.trace(C_true.T @ quat_to_dcm(q2)) - 1) / 2, -1.0, 1.0)))
    score += 1.0 if e2 < 5e-3 else 0.0
    return {"synthetic_davenport_q": score / 4.0}
