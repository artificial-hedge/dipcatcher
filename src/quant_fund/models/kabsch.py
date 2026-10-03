"""Kabsch canon: rigid (and similarity) point-set alignment —
Kabsch–Umeyama closed form via SVD with reflection correction,
recovering rotation, translation and optional scale between
corresponded point clouds. Bench: exact recovery of a known
transform, residual at machine precision, reflection handling.
All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def kabsch(
    P: FloatArray, Q: FloatArray, scale: bool = False
) -> tuple[FloatArray, FloatArray, float, float]:
    """Align P onto Q: returns (R, t, s, rms) with
    Q ≈ s·P·R + t (row-vector convention).

    With scale=False, s=1 (proper rigid fit).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d]) if P.shape[1] == 3 else np.diag([1.0, d])
    # row-vector convention: Q ≈ P·R + t with R = U·D·Vᵀ
    R = U @ D @ Vt
    var_P = float(np.sum(Pc**2))
    s = 1.0
    if scale:
        s = float(np.sum(S * np.diag(D))) / var_P if var_P > 0 else 1.0
    t = Q.mean(axis=0) - s * P.mean(axis=0) @ R
    resid = s * P @ R + t - Q
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
        s,
        rms,
    )


def apply_transform(P: FloatArray, R: FloatArray, t: FloatArray, s: float = 1.0) -> FloatArray:
    return np.asarray(s * P @ R + t, dtype=np.float64)


def bench_kabsch(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(60, 3))
    # known transform: rotation + translation
    ang = 0.6
    ca, sa = np.cos(ang), np.sin(ang)
    R_true = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    t_true = np.array([1.0, -2.0, 0.5])
    Q = P @ R_true + t_true
    R, t, s, rms = kabsch(P, Q)
    out["synthetic_kabsch_rms"] = rms
    out["synthetic_kabsch_rot_err"] = float(np.abs(R - R_true).max())
    out["synthetic_kabsch_t_err"] = float(np.abs(t - t_true).max())
    out["synthetic_kabsch_det"] = float(np.linalg.det(R))
    # noisy: rms ≈ noise level
    Qn = Q + rng.normal(0, 0.02, Q.shape)
    _, _, _, rms_n = kabsch(P, Qn)
    out["synthetic_kabsch_noisy_rms"] = rms_n
    # similarity: scale 2.5×
    Qs = 2.5 * P @ R_true + t_true
    _, _, s2, rms2 = kabsch(P, Qs, scale=True)
    out["synthetic_kabsch_scale_err"] = abs(s2 - 2.5)
    out["synthetic_kabsch_scale_rms"] = rms2
    return out
