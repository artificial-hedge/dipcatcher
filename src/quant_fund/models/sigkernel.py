"""Signature kernel via the Goursat PDE — full signature inner product (SYNTHETIC).

The (untruncated) signature kernel of two paths is the inner product of
their full signatures in the tensor algebra:

    k(a, b) = <Sig(a), Sig(b)>_{T((R^d))} = sum_{k>=0} <Sig^k(a), Sig^k(b)>.

Salvi, Chevyrev, Oberhauser & Lyons (2021), "The Signature Kernel is the
Solution of a Goursat PDE" (SIAM J. Math. Data Sci. 3(3), 873-899,
https://doi.org/10.1137/20M1366794) show that for piecewise-linear paths
the kernel satisfies

    u_{s,t}'' = <x'_s, y'_t> . u_{s,t},    u(0, t) = u(s, 0) = 1,

and discretizing on the increment grid gives the *exact* kernel value —
not a truncation: the recurrence

    k[i, j] = k[i-1, j] + k[i, j-1] + k[i-1, j-1] . (<dx_i, dy_j> - 1)
    k[0, :] = k[:, 0] = 1

is algebraically equivalent to evaluating every iterated-integral level
to infinite order for piecewise-linear input (the higher levels collapse
to a rank-one update per grid cell). This module implements the batched
form: a full Gram matrix over many path pairs is solved in one pass, with
the (i, j) grid loop carried Python-side and the pair axis vectorized.

Two-sample testing on top follows Gretton et al. (2012): the biased
V-statistic estimate of MMD^2 plus a permutation p-value, exploiting that
the pooled Gram is computed once and every permutation is then a block
sum — O(1) per permutation after one O(N^2 w^2) solve.

Anchors used by the bench:
  - exactness vs the tensor series: for sigma-scaled PL paths,
    ``sigkernel_pde`` must approach ``path_signatures.signature_kernel``'s
    truncated series as the truncation order grows — two independent
    implementations converging to the same object.
  - PSD: the Gram of any path set must be positive semidefinite
    (the kernel is an inner product by construction).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_IndexArray = NDArray[np.int64]


def _as_paths(x: Array, name: str) -> Array:
    """Coerce to a (n_paths, n_points, d) float64 array, fail closed."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim != 3:
        raise ValueError(f"{name} must be (n_points, d) or (n_paths, n_points, d); got {a.shape}")
    if a.shape[1] < 2:
        raise ValueError(f"{name} needs >= 2 points per path; got {a.shape[1]}")
    if a.shape[2] < 1:
        raise ValueError(f"{name} needs >= 1 channel; got {a.shape[2]}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return np.ascontiguousarray(a)


def sigkernel_pde(path_a: Array, path_b: Array) -> float:
    """Exact signature kernel between two piecewise-linear paths."""
    a = _as_paths(np.asarray(path_a, dtype=np.float64), "path_a")[0]
    b = _as_paths(np.asarray(path_b, dtype=np.float64), "path_b")[0]
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"paths must share the channel count; got {a.shape[1]} and {b.shape[1]}")
    gram = sigkernel_gram(a[None], b[None])
    return float(gram[0, 0])


def sigkernel_gram(x: Array, y: Array) -> Array:
    """Batched Goursat solve: Gram[i, j] = k(x_i, y_j).

    x: (nx, wx, d), y: (ny, wy, d) -> (nx, ny). The (i, j) grid recurrence
    is sequential in both axes; the (nx, ny) pair axis is vectorized and
    each increment inner product is a single matmul, so one solve costs
    (wx-1)(wy-1) small matrix multiplies instead of an O(nx ny) Gram of
    scalar solves.
    """
    xp = _as_paths(x, "x")
    yp = _as_paths(y, "y")
    if xp.shape[2] != yp.shape[2]:
        raise ValueError(f"channel mismatch: x has {xp.shape[2]}, y has {yp.shape[2]}")
    nx, wx, _ = xp.shape
    ny, wy, _ = yp.shape
    dx = np.diff(xp, axis=1)  # (nx, wx-1, d)
    dy = np.diff(yp, axis=1)  # (ny, wy-1, d)
    # K[p, q, i, j] over the (wx-1)x(wy-1) grid, borders index 0 = 1
    k = np.ones((nx, ny, wx, wy), dtype=np.float64)
    for i in range(1, wx):
        for j in range(1, wy):
            inc = dx[:, i - 1] @ dy[:, j - 1].T  # (nx, ny)
            k[:, :, i, j] = (
                k[:, :, i - 1, j] + k[:, :, i, j - 1] + k[:, :, i - 1, j - 1] * (inc - 1.0)
            )
    return k[:, :, wx - 1, wy - 1]


def mmd2(gram_xx: Array, gram_xy: Array, gram_yy: Array) -> float:
    """Biased V-statistic MMD^2 from Gram blocks."""
    gxx = np.asarray(gram_xx, dtype=np.float64)
    gxy = np.asarray(gram_xy, dtype=np.float64)
    gyy = np.asarray(gram_yy, dtype=np.float64)
    nx, ny = gxy.shape
    if gxx.shape != (nx, nx) or gyy.shape != (ny, ny):
        raise ValueError("Gram blocks inconsistent with the cross block")
    return float(gxx.mean() - 2.0 * gxy.mean() + gyy.mean())


def mmd2_paths(x: Array, y: Array) -> float:
    """Convenience: three batched solves -> MMD^2 for two path sets."""
    return mmd2(sigkernel_gram(x, x), sigkernel_gram(x, y), sigkernel_gram(y, y))


def mmd2_permutation(x: Array, y: Array, *, n_perm: int, seed: int) -> dict[str, float]:
    """Permutation two-sample test on the signature kernel.

    The pooled (n+m) Gram is computed ONCE; each permutation only re-sums
    its blocks, so the calibration cost is the Gram solve, not n_perm
    solves. Returns the observed MMD^2, the permutation p-value
    (add-one smoothed), and the permutation null's mean/std.
    """
    xp = _as_paths(x, "x")
    yp = _as_paths(y, "y")
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm}")
    nx = xp.shape[0]
    ny = yp.shape[0]
    pooled = np.concatenate([xp, yp], axis=0)
    gram = sigkernel_gram(pooled, pooled)  # (N, N)
    idx_x = np.arange(nx, dtype=np.int64)
    idx_y = np.arange(nx, nx + ny, dtype=np.int64)

    def _mmd(ix: _IndexArray, iy: _IndexArray) -> float:
        return mmd2(gram[np.ix_(ix, ix)], gram[np.ix_(ix, iy)], gram[np.ix_(iy, iy)])

    observed = _mmd(idx_x, idx_y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    all_idx = np.arange(nx + ny, dtype=np.int64)
    for b in range(n_perm):
        perm = rng.permutation(all_idx)
        null[b] = _mmd(np.sort(perm[:nx]), np.sort(perm[nx:]))
    p = float((1.0 + np.count_nonzero(null >= observed - 1e-12)) / (n_perm + 1.0))
    return {
        "mmd2": observed,
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=0)),
        "n_perm": float(n_perm),
    }


def gram_is_psd(gram: Array, *, tol: float = -1e-8) -> bool:
    """PSD certificate: smallest eigenvalue not below tolerance."""
    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("gram must be square")
    if not np.isfinite(g).all():
        return False
    w = np.linalg.eigvalsh((g + g.T) * 0.5)
    return bool(w.min() > tol)


def truncated_oracle(path_a: Array, path_b: Array, *, order: int, sigma: float) -> float:
    """The tensor-series oracle: truncated inner product via path_signatures.

    Isolated import so ``sigkernel_pde`` stays dependency-free; the bench
    uses this as the independent implementation the PDE solve must match
    in the small-increment regime.
    """
    from quant_fund.models.path_signatures import signature_kernel

    return signature_kernel(path_a, path_b, order=order, sigma=sigma)


def pde_vs_series_gap(
    path_a: Array, path_b: Array, *, sigmas: tuple[float, ...]
) -> dict[str, float]:
    """|k_pde - k_series6| across sigma scales; must shrink as sigma -> 0."""
    gaps: dict[str, float] = {}
    for s in sigmas:
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError(f"sigma must be positive and finite; got {s!r}")
        a = np.asarray(path_a, dtype=np.float64) * s
        b = np.asarray(path_b, dtype=np.float64) * s
        k_pde = sigkernel_pde(a, b)
        k_ser = truncated_oracle(a, b, order=6, sigma=1.0)
        gaps[f"{s}"] = abs(k_pde - k_ser)
    return gaps
