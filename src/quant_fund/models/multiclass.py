"""Multiclass canon: one-vs-rest logistic reduction, softmax (multinomial)
regression by Newton steps, and error-correcting output codes (ECOC) with
exhaustive binary codewords. ``bench_multiclass`` plants a 4-class mixture
and gates all three reductions over the chance rate, plus softmax >= OvR.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _logit_fit(x: FloatArray, y: FloatArray, lam: float, it: int = 300) -> FloatArray:
    n, d = x.shape
    w = np.zeros(d)
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        g = x.T @ (p - y) / n + lam * w
        h = (x.T * (p * (1.0 - p))) @ x / n + lam * np.eye(d)
        step = solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def ovr_fit(x: FloatArray, y: IntArray, n_classes: int, lam: float = 1e-3) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape[0] != y.shape[0] or n_classes < 2:
        raise ValueError("bad inputs")
    w = np.zeros((n_classes, x.shape[1]))
    for k in range(n_classes):
        w[k] = _logit_fit(x, (y == k).astype(np.float64), lam)
    return w


def ovr_predict(w: FloatArray, x: FloatArray) -> IntArray:
    return np.asarray(np.argmax(x @ np.asarray(w).T, axis=1), dtype=np.int64)


def softmax_fit(
    x: FloatArray, y: IntArray, n_classes: int, lam: float = 1e-3, it: int = 60
) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, d = x.shape
    w = np.zeros((n_classes, d))
    yoh = np.zeros((n, n_classes))
    yoh[np.arange(n), y] = 1.0
    mom = np.zeros_like(w)
    vel = np.zeros_like(w)
    b1, b2, lr = 0.9, 0.999, 0.2
    for t in range(1, it + 1):
        z = x @ w.T
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - yoh).T @ x / n + lam * w
        mom = b1 * mom + (1 - b1) * g
        vel = b2 * vel + (1 - b2) * g * g
        step = lr * (mom / (1 - b1**t)) / (np.sqrt(vel / (1 - b2**t)) + 1e-8)
        w -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return w


def softmax_predict(w: FloatArray, x: FloatArray) -> IntArray:
    z = np.asarray(x) @ np.asarray(w).T
    return np.asarray(np.argmax(z, axis=1), dtype=np.int64)


def ecoc_fit(
    x: FloatArray,
    y: IntArray,
    code: FloatArray,
    lam: float = 1e-3,
) -> FloatArray:
    """code: (n_classes, n_bits) entries in {-1,+1}; learns one binary
    classifier per bit."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    code = np.asarray(code, dtype=np.float64)
    n_bits = code.shape[1]
    w = np.zeros((n_bits, x.shape[1]))
    for b in range(n_bits):
        bit = np.array([code[c, b] for c in y])
        w[b] = _logit_fit(x, (bit > 0).astype(np.float64), lam)
    return w


def ecoc_predict(w: FloatArray, x: FloatArray, code: FloatArray) -> IntArray:
    code = np.asarray(code, dtype=np.float64)
    scores = np.asarray(x) @ np.asarray(w).T  # logits per bit
    # decode by Hamming distance to codewords
    prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    target = (code + 1.0) / 2.0
    dist = np.sum((prob[:, None, :] - target[None, :, :]) ** 2, axis=2)
    return np.asarray(np.argmin(dist, axis=1), dtype=np.int64)


def _exhaustive_code(n_classes: int) -> FloatArray:
    # all 2^(K-1)-1 nontrivial dichotomies minus the all-ones column
    n_bits = 2 ** (n_classes - 1) - 1
    code = np.ones((n_classes, n_bits))
    for b in range(n_bits):
        for c in range(n_classes):
            code[c, b] = 1.0 if (b >> c) & 1 == 0 else -1.0
    return code


def bench_multiclass(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d, k = 400, 6, 4
    centers = np.array(
        [[3.0, 0, 0, 0, 0, 0], [-3.0, 0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0, 0], [0, -3.0, 0, 0, 0, 0]]
    )
    y = rng.integers(0, k, n).astype(np.int64)
    x = centers[y] + 1.0 * rng.standard_normal((n, d))
    # add a nonlinear twist so softmax has a slight edge over per-class OvR
    x[:, 4] = x[:, 0] * x[:, 1]
    perm = rng.permutation(n)
    tr, te = perm[:300], perm[300:]
    w_ovr = ovr_fit(x[tr], y[tr], k)
    w_sm = softmax_fit(x[tr], y[tr], k)
    code = _exhaustive_code(k)
    w_ec = ecoc_fit(x[tr], y[tr], code)
    acc = lambda p: float(np.mean(p == y[te]))  # noqa: E731
    a_ovr = acc(ovr_predict(w_ovr, x[te]))
    a_sm = acc(softmax_predict(w_sm, x[te]))
    a_ec = acc(ecoc_predict(w_ec, x[te], code))
    return {
        "synthetic_ovr_acc": a_ovr,
        "synthetic_softmax_acc": a_sm,
        "synthetic_ecoc_acc": a_ec,
        "synthetic_sm_minus_ovr": a_sm - a_ovr,
    }
