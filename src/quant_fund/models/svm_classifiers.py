"""Support-vector machines: Pegasos primal subgradient
descent for linear soft-margin SVMs (Shalev-Shwartz et al.
2011), and kernel SVM via pair-free coordinate ascent on the
dual (kernel Pegasos, α ∈ [0, 1/λn]). Synthetic bench gates
separable-blob margins and circle-vs-linear accuracy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pegasos_fit(
    x: FloatArray, y: FloatArray, lam: float = 0.01, it: int = 2000, seed: int = 0
) -> FloatArray:
    """w ← (1−ηλ)w + η·y_i x_i on hinge violations, η=1/(λt).
    A constant column carries the intercept."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64) * 2 - 1  # {0,1} → {−1,+1}
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    for t in range(1, it + 1):
        i = int(rng.integers(n))
        eta = 1.0 / (lam * t)
        w *= 1 - eta * lam
        if y[i] * float(x[i] @ w) < 1.0:
            w += eta * y[i] * x[i]
        # project onto the 1/sqrt(lam) ball
        nw = np.linalg.norm(w)
        if nw > 1 / np.sqrt(lam):
            w /= nw * np.sqrt(lam)
    return np.asarray(w)


def linear_svm_predict(w: FloatArray, x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if w.shape[0] == x.shape[1] + 1:
        x = np.c_[x, np.ones(len(x))]
    return np.asarray((x @ w > 0).astype(np.float64))


def _rbf(x: FloatArray, z: FloatArray, gamma: float) -> FloatArray:
    d2 = ((x[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(np.exp(-gamma * d2))


def kernel_svm_fit(
    x: FloatArray,
    y: FloatArray,
    lam: float = 0.01,
    gamma: float = 1.0,
    it: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Kernel Pegasos: α_i updates on dual objective; decision
    f(x) = Σ_i α_i y_i K(x_i, x)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64) * 2 - 1
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    k = _rbf(x, x, gamma)
    alpha = np.zeros(n)
    for t in range(1, it + 1):
        i = int(rng.integers(n))
        eta = 1.0 / (lam * t)
        f_i = float((alpha * y * k[:, i]).sum())
        alpha *= 1 - eta * lam
        if y[i] * f_i < 1.0:
            alpha[i] += eta
    return {"alpha": alpha, "x": x, "y": y, "gamma": gamma}


def kernel_svm_predict(model: dict[str, object], x_new: FloatArray) -> FloatArray:
    x = np.asarray(model["x"])
    y = np.asarray(model["y"])
    alpha = np.asarray(model["alpha"])
    gamma = float(np.asarray(model["gamma"]))
    x_new = np.asarray(x_new, dtype=np.float64)
    k = _rbf(x_new, x, gamma)
    f = k @ (alpha * y)
    return np.asarray((f > 0).astype(np.float64))


def bench_svm(seed: int = 552) -> dict[str, float]:
    """SYNTHETIC: linear SVM separates blobs with a wide
    margin; kernel SVM fits a circle where linear cannot."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # blobs: margin achieved
    x = np.vstack([rng.normal([0, 0], 0.4, (100, 2)), rng.normal([3, 3], 0.4, (100, 2))])
    y = np.r_[np.zeros(100), np.ones(100)]
    w = pegasos_fit(x, y, lam=0.01, it=3000, seed=seed)
    acc_lin = float((linear_svm_predict(w, x) == y).mean())
    out["synthetic_pegasos_blob_acc"] = acc_lin
    # circle: kernel SVM vs linear control
    t = rng.uniform(0, 2 * np.pi, 240)
    r_in = rng.uniform(0, 1, 120)
    r_out = rng.uniform(2, 3, 120)
    xc = np.vstack(
        [
            np.c_[r_in * np.cos(t[:120]), r_in * np.sin(t[:120])],
            np.c_[r_out * np.cos(t[:120]), r_out * np.sin(t[:120])],
        ]
    )
    yc = np.r_[np.zeros(120), np.ones(120)]
    mdl = kernel_svm_fit(xc, yc, lam=0.01, gamma=0.8, it=4000, seed=seed)
    acc_k = float((kernel_svm_predict(mdl, xc) == yc).mean())
    out["synthetic_ksvm_circle_acc"] = acc_k
    wl = pegasos_fit(xc, yc, lam=0.01, it=2000, seed=seed + 1)
    acc_l = float((linear_svm_predict(wl, xc) == yc).mean())
    out["synthetic_linear_circle_acc"] = acc_l
    if acc_lin < 0.95:
        raise ValueError(f"pegasos blob acc off: {acc_lin}")
    if acc_k < 0.9:
        raise ValueError(f"ksvm circle acc off: {acc_k}")
    if acc_k < acc_l + 0.15:
        raise ValueError(f"ksvm not >> linear: {acc_k} vs {acc_l}")
    return out
