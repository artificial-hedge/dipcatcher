"""Boosting: AdaBoost.M1 (Freund-Schapire 1997) and
LogitBoost (Friedman-Hastie-Tibshirani 2000 — additive
logistic regression by Newton steps), both on decision
stumps. Synthetic bench gates boosted accuracy over a
single stump on a margin fixture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _stump(x: FloatArray, y: FloatArray, w: FloatArray) -> tuple[int, float, float, float]:
    """Best weighted decision stump (feature, threshold,
    left-val, right-val) — exhaustive thresholds."""
    best = (0, 0.0, -1.0, 1.0)
    best_err = np.inf
    n, d = x.shape
    for j in range(d):
        order = np.argsort(x[:, j])
        xs = x[order, j]
        ys = y[order]
        ws = w[order]
        for t in np.unique((xs[:-1] + xs[1:]) / 2):
            left = xs <= t
            # weighted majority on each side
            for lv, rv in ((-1, 1), (1, -1)):
                pred = np.where(left, lv, rv)
                err = float(ws[pred != ys].sum())
                if err < best_err:
                    best_err = err
                    best = (j, float(t), float(lv), float(rv))
    return best


def _stump_pred(x: FloatArray, stump: tuple[int, float, float, float]) -> FloatArray:
    j, t, lv, rv = stump
    return np.asarray(np.where(x[:, j] <= t, lv, rv))


def adaboost_fit(x: FloatArray, y: FloatArray, it: int = 50) -> dict[str, object]:
    """AdaBoost.M1: reweight misclassified, α = ½ ln((1−ε)/ε)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y) * 2 - 1
    w = np.full(len(y), 1.0 / len(y))
    stumps: list[tuple[int, float, float, float]] = []
    alphas: list[float] = []
    for _ in range(it):
        st = _stump(x, y, w)
        pred = _stump_pred(x, st)
        err = float(w[pred != y].sum())
        if err >= 0.5 - 1e-9 or err <= 1e-12:
            if err <= 1e-12:
                stumps.append(st)
                alphas.append(5.0)
            break
        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(-alpha * y * pred)
        w /= w.sum()
        stumps.append(st)
        alphas.append(alpha)
    return {"stumps": stumps, "alphas": alphas}


def adaboost_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    stumps = model["stumps"]
    alphas = model["alphas"]
    if not (isinstance(stumps, list) and isinstance(alphas, list)):
        raise ValueError("isinstance(stumps, list) and isinstance(alphas, list)")
    s = np.zeros(len(x))
    for st, a in zip(stumps, alphas, strict=True):
        s += a * _stump_pred(x, st)
    return np.asarray((s > 0).astype(np.float64))


def logitboost_fit(
    x: FloatArray, y: FloatArray, it: int = 30, nu: float = 0.5
) -> dict[str, object]:
    """LogitBoost: fit stumps to z = (y−p)/(p(1−p)) weighted
    by p(1−p) — simpler working-response variant."""
    x = np.asarray(x, dtype=np.float64)
    f = np.zeros(len(y))
    stumps: list[tuple[int, float, float, float]] = []
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-f))
        z = (np.asarray(y) - p) / np.maximum(p * (1 - p), 1e-6)
        zw = p * (1 - p)
        # weighted stump regression on z: pick feature/threshold
        # minimizing weighted SSE; predict weighted means
        best = (0, 0.0, 0.0, 0.0)
        best_sse = np.inf
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j])
            xs = x[order, j]
            for t in np.unique((xs[:-1] + xs[1:]) / 2):
                left = xs <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                wl, wr = zw[order][left], zw[order][~left]
                zl, zr = z[order][left], z[order][~left]
                ml = float((wl * zl).sum() / max(wl.sum(), 1e-12))
                mr = float((wr * zr).sum() / max(wr.sum(), 1e-12))
                sse = float((wl * (zl - ml) ** 2).sum() + (wr * (zr - mr) ** 2).sum())
                if sse < best_sse:
                    best_sse = sse
                    best = (j, float(t), ml, mr)
        j, t, ml, mr = best
        upd = np.where(x[:, j] <= t, ml, mr)
        f += nu * upd
        stumps.append(best)
    return {"stumps": stumps, "nu": nu}


def logitboost_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    stumps = model["stumps"]
    nu = float(np.asarray(model["nu"]))
    if not (isinstance(stumps, list)):
        raise ValueError("isinstance(stumps, list)")
    f = np.zeros(len(x))
    for j, t, ml, mr in stumps:
        f += nu * np.where(x[:, j] <= t, ml, mr)
    return np.asarray((f > 0).astype(np.float64))


def bench_adaboost(seed: int = 565) -> dict[str, float]:
    """SYNTHETIC: margins fixture where no single axis
    stump separates well — boosting must beat the best
    stump materially."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 400
    x = rng.normal(0, 1, (n, 4))
    # class boundary = rotated hyperplane: no axis stump is great
    u = np.ones(4) / 2
    y = (x @ u + rng.normal(0, 0.5, n) > 0).astype(np.float64)
    perm = rng.permutation(n)
    tr, te = perm[:300], perm[300:]
    # single stump baseline
    st = _stump(x[tr], y[tr] * 2 - 1, np.full(len(tr), 1 / len(tr)))
    acc_stump = float((_stump_pred(x[te], st) == y[te] * 2 - 1).mean())
    out["synthetic_stump_acc"] = acc_stump
    m = adaboost_fit(x[tr], y[tr], it=60)
    acc_ada = float((adaboost_predict(m, x[te]) == y[te]).mean())
    out["synthetic_adaboost_acc"] = acc_ada
    if acc_ada < acc_stump + 0.05:
        raise ValueError(f"adaboost not > stump: {acc_ada} vs {acc_stump}")
    ml = logitboost_fit(x[tr], y[tr], it=40, nu=0.5)
    acc_lb = float((logitboost_predict(ml, x[te]) == y[te]).mean())
    out["synthetic_logitboost_acc"] = acc_lb
    if acc_lb < acc_stump + 0.03:
        raise ValueError(f"logitboost not > stump: {acc_lb} vs {acc_stump}")
    return out
