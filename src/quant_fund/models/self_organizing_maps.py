"""Kohonen self-organizing maps and learning vector (SYNTHETIC)
quantization: incremental SOM with Gaussian neighborhood
decay, quantization/topographic error, U-matrix, and LVQ1
classification. Synthetic bench gates topology preservation
and class separation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _grid_coords(rows: int, cols: int) -> FloatArray:
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.asarray(np.c_[r.ravel(), c.ravel()], dtype=np.float64)


def som_train(
    x: FloatArray,
    rows: int,
    cols: int,
    it: int = 500,
    lr0: float = 0.5,
    sigma0: float | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Incremental SOM (Kohonen 1982): BMU argmin + Gaussian
    neighborhood update; lr and σ decay linearly to ~1% of
    initial."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    w = x[rng.integers(0, n, rows * cols)].copy()
    coords = _grid_coords(rows, cols)
    if sigma0 is None:
        sigma0 = max(rows, cols) / 2.0
    for t in range(it):
        frac = t / max(it - 1, 1)
        lr = lr0 * (1 - frac) + 0.01 * lr0
        sigma = sigma0 * (1 - frac) + 0.05
        xi = x[rng.integers(n)]
        j = int(np.argmin(((w - xi) ** 2).sum(axis=1)))
        h = np.exp(-(((coords - coords[j]) ** 2).sum(axis=1)) / (2 * sigma**2))
        w += lr * h[:, None] * (xi - w)
    return {"weights": w, "coords": coords, "shape": (rows, cols)}


def som_bmus(w: FloatArray, x: FloatArray) -> FloatArray:
    """Best-matching-unit index per sample."""
    x = np.asarray(x, dtype=np.float64)
    d2 = ((x[:, None, :] - w[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(np.argmin(d2, axis=1))


def quantization_error(w: FloatArray, x: FloatArray) -> float:
    """Mean distance to BMU."""
    x = np.asarray(x, dtype=np.float64)
    d2 = ((x[:, None, :] - w[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).mean())


def topographic_error(w: FloatArray, coords: FloatArray, x: FloatArray) -> float:
    """Fraction of samples whose first and second BMUs are
    not lattice-adjacent (Vilmann et al.)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    errs = 0
    for xi in x:
        d2 = ((w - xi) ** 2).sum(axis=1)
        b1, b2 = np.argsort(d2)[:2]
        adj = np.abs(coords[b1] - coords[b2]).sum() <= 1.0
        if not adj:
            errs += 1
    return float(errs / n)


def u_matrix(w: FloatArray, rows: int, cols: int) -> FloatArray:
    """U-matrix: mean neighbor distance per unit."""
    w = np.asarray(w, dtype=np.float64)
    w = w.reshape(rows, cols, -1)
    u = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            d = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < rows and 0 <= jj < cols:
                    d.append(float(np.linalg.norm(w[i, j] - w[ii, jj])))
            u[i, j] = np.mean(d) if d else 0.0
    return u


def lvq_train(
    x: FloatArray,
    y: FloatArray,
    n_proto: int = 3,
    it: int = 200,
    lr0: float = 0.3,
    seed: int = 0,
) -> dict[str, object]:
    """LVQ1 (Kohonen 1988): prototypes per class updated
    toward/away from the sample by winner-take-all sign."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    protos = []
    plabels = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        pick = idx[rng.integers(0, len(idx), n_proto)]
        protos.append(x[pick])
        plabels.append(np.full(n_proto, c))
    w = np.vstack(protos)
    wl = np.concatenate(plabels)
    for t in range(it):
        lr = lr0 * (1 - t / it)
        i = rng.integers(len(x))
        j = int(np.argmin(((w - x[i]) ** 2).sum(axis=1)))
        sign = 1.0 if wl[j] == y[i] else -1.0
        w[j] += lr * sign * (x[i] - w[j])
    return {"protos": w, "labels": wl}


def lvq_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    """Nearest-prototype class label."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(model["protos"])
    wl = np.asarray(model["labels"])
    d2 = ((x[:, None, :] - w[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(wl[np.argmin(d2, axis=1)])


def bench_som(seed: int = 543) -> dict[str, float]:
    """SYNTHETIC: ring + disk clusters — SOM QE shrinks vs an
    untrained map, topographic error bounded; LVQ classifies
    held-out samples near-perfectly."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 300
    inner = rng.normal(scale=0.4, size=(n // 2, 2))
    theta = rng.uniform(0, 2 * np.pi, n // 2)
    outer = np.c_[3 * np.cos(theta), 3 * np.sin(theta)] + rng.normal(scale=0.25, size=(n // 2, 2))
    x = np.vstack([inner, outer])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(np.int64)
    r = som_train(x, 8, 8, it=1500, lr0=0.7, seed=seed)
    w = np.asarray(r["weights"])
    coords = np.asarray(r["coords"])
    qe = quantization_error(w, x)
    te = topographic_error(w, coords, x)
    out["synthetic_som_qe"] = qe
    out["synthetic_som_te"] = te
    # uniform-random prototype QE for contrast (best of 5 draws)
    rng_r = np.random.default_rng(seed + 1)
    qe0 = min(quantization_error(rng_r.uniform(x.min(0), x.max(0), (64, 2)), x) for _ in range(5))
    out["synthetic_som_qe_random"] = qe0
    if qe >= qe0 * 0.9:
        raise ValueError(f"som qe not improved: {qe} vs {qe0}")
    if te > 0.2:
        raise ValueError(f"topographic error off: {te}")
    # LVQ on two-moons-shaped parity classes
    mdl = lvq_train(x, y, n_proto=4, it=400, seed=seed)
    pred = lvq_predict(mdl, x)
    acc = float((pred == y).mean())
    out["synthetic_lvq_acc"] = acc
    if acc < 0.9:
        raise ValueError(f"lvq acc off: {acc}")
    return out
