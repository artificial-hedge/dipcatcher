"""Fuzzy clustering: Bezdek fuzzy c-means with validity
indices — partition coefficient, partition entropy, and the
Xie–Beni compactness/separation index. Synthetic bench gates
cluster recovery and validity ordering on planted blobs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fuzzy_cmeans(
    x: FloatArray,
    k: int,
    m: float = 2.0,
    it: int = 150,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict[str, object]:
    """FCM (Bezdek 1981): memberships u_ij ∝ d_ij^{-2/(m-1)},
    centers v_j = Σ_i u_ij^m x_i / Σ_i u_ij^m."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    u = rng.uniform(size=(n, k))
    u /= u.sum(axis=1, keepdims=True)
    prev_obj = np.inf
    obj = np.inf
    for _ in range(it):
        v = (u**m).T @ x / np.maximum((u**m).sum(axis=0), 1e-12)[:, None]
        d = np.linalg.norm(x[:, None, :] - v[None, :, :], axis=2)
        d = np.maximum(d, 1e-12)
        p = 2.0 / (m - 1.0)
        ratio = (d[:, :, None] / d[:, None, :]) ** p
        u = 1.0 / ratio.sum(axis=2)
        obj = float(((u**m) * (d**2)).sum())
        if abs(prev_obj - obj) < tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj
    labels = np.argmax(u, axis=1)
    return {"centers": v, "membership": u, "labels": labels, "objective": obj}


def partition_coefficient(u: FloatArray) -> float:
    """PC = (1/n)Σ_ij u_ij²  ∈ [1/k, 1]; higher = crisper."""
    u = np.asarray(u, dtype=np.float64)
    return float((u**2).sum() / u.shape[0])


def partition_entropy(u: FloatArray) -> float:
    """PE = -(1/n)Σ_ij u_ij log u_ij; lower = crisper."""
    u = np.asarray(u, dtype=np.float64)
    lu = np.log(np.maximum(u, 1e-16))
    return float(-(u * lu).sum() / u.shape[0])


def xie_beni(x: FloatArray, u: FloatArray, v: FloatArray, m: float = 2.0) -> float:
    """XB = J_m / (n · min_{j≠l}‖v_j − v_l‖²); lower = better."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d2 = ((x[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
    jm = float(((u**m) * d2).sum())
    dv = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    np.fill_diagonal(dv, np.inf)
    sep = float(dv.min() ** 2)
    return float(jm / (x.shape[0] * max(sep, 1e-16)))


def bench_fuzzy(seed: int = 542) -> dict[str, float]:
    """SYNTHETIC: three well-separated blobs — FCM recovers
    centers, PC near 1, XB small vs a scrambled-membership
    control."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    blobs = np.vstack(
        [
            rng.normal([0, 0], 0.25, (60, 2)),
            rng.normal([5, 5], 0.25, (60, 2)),
            rng.normal([0, 6], 0.25, (60, 2)),
        ]
    )
    truth = np.repeat([0, 1, 2], 60)
    r = fuzzy_cmeans(blobs, 3, seed=seed)
    u = np.asarray(r["membership"])
    v = np.asarray(r["centers"])
    lab = np.asarray(r["labels"])
    # match labels by center proximity
    centers_true = np.array([[0, 0], [5, 5], [0, 6]])
    perm = np.array([int(np.argmin(((centers_true - c) ** 2).sum(axis=1))) for c in v])
    acc = float((perm[lab] == truth).mean())
    out["synthetic_fcm_acc"] = acc
    if acc < 0.97:
        raise ValueError(f"fcm acc off: {acc}")
    pc = partition_coefficient(u)
    pe = partition_entropy(u)
    xb = xie_beni(blobs, u, v)
    out["synthetic_fcm_pc"] = pc
    out["synthetic_fcm_pe"] = pe
    out["synthetic_fcm_xb"] = xb
    if pc < 0.95:
        raise ValueError(f"pc off: {pc}")
    # scrambled memberships must look worse on both indices
    u_scr = rng.uniform(size=u.shape)
    u_scr /= u_scr.sum(axis=1, keepdims=True)
    out["synthetic_fcm_pc_scr"] = partition_coefficient(u_scr)
    if not pc > partition_coefficient(u_scr):
        raise ValueError("pc ordering off")
    if not pe < partition_entropy(u_scr):
        raise ValueError("pe ordering off")
    return out
