"""Kernel-Stein thinning and kernel herding for MCMC sample compression.

Posterior samplers emit thousands of correlated draws; downstream
consumers (benchmarks, stress suites, ensembling) want a small coreset
that still represents the posterior. Two complementary compressors:

- **Kernel herding** (Chen, Welling & Smola 2010): greedy maximum-a-
  posteriori subset selection that greedily minimizes the MMD between
  the retained subset and the full sample — deterministic super-samples
  that beat random subsampling.
- **Stein thinning** (Riabiz et al. 2022): greedy minimization of the
  *kernel Stein discrepancy* (KSD) of the retained set. The KSD uses the
  target's score function ``s(x) = ∇ log p(x)``, so thinning respects
  the posterior geometry rather than only the sample geometry. The
  objective implemented is the empirical KSD of the retained set,
  ``KSD²(S) = (1/|S|²) Σ_{i,j ∈ S} k₀(x_i, x_j)`` with ``k₀`` the
  RBF-based Langevin-Stein kernel — removal steps are closed form via
  Gram row-sums (the "simple greedy" variant documented below).

Functions
---------
- :func:`median_bandwidth` — median-heuristic RBF length-scale.
- :func:`rbf_gram` — RBF Gram matrix ``exp(-‖x-y‖² / 2 bw²)``.
- :func:`mmd_rbf` — biased V-statistic MMD² between two samples.
- :func:`kernel_herding` — greedy MMD-minimizing subset indices.
- :func:`stein_kernel_gram` — Langevin-Stein kernel Gram from a score.
- :func:`stein_thin` — greedy KSD-minimizing thinning to ``m`` points.
- :func:`compress_report` — thinned vs random-subset MMD comparison.
- :func:`synth_posterior` — correlated-Gaussian MCMC stand-in.
- :func:`bench_stein_thinning` — SYNTHETIC telemetry blob.

References
----------
- Riabiz, Chen, Cockayne, Swietach, Niederer, Mackey & Oates (2022).
  Optimal thinning of MCMC output. *JRSS-B* 84 — arXiv:2005.03952.
- Teymur, Gorham, Riabiz & Oates (2021). Optimal quantisation of
  probability measures using maximum mean discrepancy. *AISTATS* —
  kernel thinning, arXiv:2105.05842.
- Chen, Welling & Smola (2010). Super-samples from kernel herding.
  *UAI* — arXiv:1203.3472.
- Liu, Lee & Jordan (2016). A kernelized Stein discrepancy for
  goodness-of-fit tests. *ICML* — arXiv:1602.03253.
- Gorham & Mackey (2017). Measuring sample quality with kernels.
  *ICML* — arXiv:1703.01717.

Honesty
-------
All reported numbers are SYNTHETIC compression checks on seeded
correlated-Gaussian stand-ins — they validate the thinning machinery,
never real posteriors or market data. MMD/KSD improvements are
distributional-similarity diagnostics, not forecasting claims.

Composition notes
-----------------
- ``metrics/kernel_changepoint.py`` (wave 25): same RBF/median-heuristic
  machinery applied to two-sample changepoint detection — that module
  lives on its canon branch, so the biased-V MMD is reimplemented here
  rather than imported.
- ``metrics/wasserstein.py``: OT-based sample comparator — MMD here is
  the kernel-space complement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _check_xy(x: FloatArray, n_min: int = 2) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < n_min or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n>=2, d) array")
    return arr


def median_bandwidth(x: FloatArray) -> float:
    """Median of pairwise Euclidean distances (RBF length-scale heuristic)."""
    arr = _check_xy(x)
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    iu = np.triu_indices(arr.shape[0], k=1)
    med = float(np.median(dist[iu]))
    if not np.isfinite(med) or med <= 0.0:
        raise ValueError("median bandwidth is degenerate (identical points?)")
    return med


def rbf_gram(x: FloatArray, y: FloatArray, bw: float) -> FloatArray:
    """RBF Gram ``K[i,j] = exp(-‖x_i - y_j‖² / (2 bw²))``."""
    xa = _check_xy(x, n_min=1)
    ya = _check_xy(y, n_min=1)
    if xa.shape[1] != ya.shape[1] or bw <= 0 or not np.isfinite(bw):
        raise ValueError("dim mismatch or non-positive bandwidth")
    sq = ((xa[:, None, :] - ya[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-sq / (2.0 * bw * bw))


def mmd_rbf(x: FloatArray, y: FloatArray, bw: float | None = None) -> float:
    """Biased V-statistic MMD² with the RBF kernel (includes diagonals)."""
    xa = _check_xy(x)
    ya = _check_xy(y)
    if bw is None:
        bw = median_bandwidth(np.vstack([xa, ya]))
    kxx = float(rbf_gram(xa, xa, bw).mean())
    kyy = float(rbf_gram(ya, ya, bw).mean())
    kxy = float(rbf_gram(xa, ya, bw).mean())
    return max(kxx + kyy - 2.0 * kxy, 0.0)


def kernel_herding(x: FloatArray, m: int, bw: float | None = None) -> IntArray:
    """Greedy kernel-herding subset: indices minimizing subset-to-sample MMD.

    Each step adds ``argmax_i [ mean_j k(x_i, x_j) - mean_{s∈S} k(x_i, x_s) ]`` —
    the classic herding trade-off between representing the sample and not
    duplicating already-selected points.
    """
    arr = _check_xy(x)
    n = arr.shape[0]
    if not 1 <= m <= n:
        raise ValueError("m must be in [1, n]")
    if bw is None:
        bw = median_bandwidth(arr)
    gram = rbf_gram(arr, arr, bw)
    target = gram.mean(axis=1)  # mean_j k(x_i, x_j)
    selected: list[int] = []
    penalty = np.zeros(n)
    for _ in range(m):
        score = target - penalty
        if selected:
            score[np.asarray(selected)] = -np.inf
        nxt = int(np.argmax(score))
        selected.append(nxt)
        penalty = penalty + (gram[:, nxt] - penalty) / len(selected)
    return np.asarray(selected, dtype=np.int_)


def stein_kernel_gram(x: FloatArray, score: FloatArray, bw: float) -> FloatArray:
    """Langevin-Stein kernel Gram for the RBF base kernel.

    ``k₀(x,y) = s_x·s_y·k + s_x·∇_y k + s_y·∇_x k + ∇_x·∇_y k`` which for
    ``k = exp(-‖x-y‖²/2bw²)`` evaluates to
    ``k·[s_x·s_y + (s_x - s_y)·(x - y)/bw² + d/bw² - ‖x-y‖²/bw⁴]``.
    """
    arr = _check_xy(x)
    sc = np.asarray(score, dtype=np.float64)
    if sc.shape != arr.shape or not np.isfinite(sc).all():
        raise ValueError("score must be finite with shape (n, d) matching x")
    if bw <= 0 or not np.isfinite(bw):
        raise ValueError("non-positive bandwidth")
    n, d = arr.shape
    k = rbf_gram(arr, arr, bw)
    diff = arr[:, None, :] - arr[None, :, :]  # x_i - x_j
    sq = (diff**2).sum(axis=2)
    ss = sc @ sc.T
    sdiff = sc[:, None, :] - sc[None, :, :]  # s_i - s_j
    cross = (sdiff * diff).sum(axis=2) / (bw * bw)
    lap = d / (bw * bw) - sq / (bw**4)
    return np.asarray(k * (ss + cross + lap), dtype=np.float64)


def stein_thin(
    x: FloatArray,
    score: FloatArray,
    m: int,
    bw: float | None = None,
) -> IntArray:
    """Greedy KSD thinning: iteratively remove the point whose deletion most
    decreases ``KSD²(S) = (1/|S|²) Σ_{i,j∈S} k₀_ij``.

    Removing ``r`` changes the Gram total from ``T`` to
    ``T - 2·row_sum_r + k₀_rr`` and the normalizer from ``n²`` to
    ``(n-1)²`` — each step evaluates all candidates in closed form and
    deletes the argmin. This is the documented simple greedy variant of
    Stein thinning (Riabiz et al. 2022 §4).
    """
    arr = _check_xy(x)
    n = arr.shape[0]
    if not 1 <= m <= n:
        raise ValueError("m must be in [1, n]")
    if bw is None:
        bw = median_bandwidth(arr)
    k0 = stein_kernel_gram(arr, score, bw)
    active = np.ones(n, dtype=bool)
    for _ in range(n - m):
        sub = k0[np.ix_(active, active)]
        cur_n = sub.shape[0]
        total = float(sub.sum())
        row_sum = sub.sum(axis=1)
        diag = np.diag(sub)
        # KSD² after removing candidate r
        new_total = total - 2.0 * row_sum + diag
        ksd_after = new_total / ((cur_n - 1.0) ** 2)
        drop_local = int(np.argmin(ksd_after))
        idx = np.flatnonzero(active)[drop_local]
        active[idx] = False
    return np.flatnonzero(active)


def compress_report(
    x: FloatArray,
    m: int,
    bw: float | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """MMD of the herding-thinned subset vs an equal-size random subset."""
    arr = _check_xy(x)
    n = arr.shape[0]
    if not 1 <= m < n:
        raise ValueError("m must be in [1, n)")
    if bw is None:
        bw = median_bandwidth(arr)
    herd = kernel_herding(arr, m, bw)
    rng = np.random.default_rng(seed)
    rand = rng.choice(n, size=m, replace=False)
    mmd_herd = mmd_rbf(arr[herd], arr, bw)
    mmd_rand = mmd_rbf(arr[rand], arr, bw)
    return {
        "mmd_herd": mmd_herd,
        "mmd_random": mmd_rand,
        "mmd_edge": mmd_herd / mmd_rand if mmd_rand > 0 else 1.0,
    }


def synth_posterior(n: int, dim: int, corr: float, seed: int) -> tuple[FloatArray, FloatArray]:
    """Correlated-Gaussian MCMC stand-in; returns ``(samples, scores)``."""
    if n < 8 or dim < 1 or not -0.95 < corr < 0.95:
        raise ValueError("bad n/dim/corr")
    rng = np.random.default_rng(seed)
    cov = np.full((dim, dim), corr) + np.eye(dim) * (1.0 - corr)
    chol = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, dim)) @ chol.T
    # score of N(0, cov): s = -cov^{-1} x
    score = -np.linalg.solve(cov, x.T).T
    return x, score


def bench_stein_thinning(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC thinning quality blob — compression diagnostics only."""
    rng = np.random.default_rng(seed)
    x, score = synth_posterior(160, 4, 0.6, seed)
    bw = median_bandwidth(x)
    m = 32

    thin_idx = stein_thin(x, score, m, bw)
    herd_idx = kernel_herding(x, m, bw)
    rand_idx = rng.choice(x.shape[0], size=m, replace=False)

    k0 = stein_kernel_gram(x, score, bw)

    def ksd(idxa: IntArray) -> float:
        sub = k0[np.ix_(idxa, idxa)]
        return float(max(sub.mean(), 0.0))

    mmd_thin = mmd_rbf(x[thin_idx], x, bw)
    mmd_herd = mmd_rbf(x[herd_idx], x, bw)
    mmd_rand = mmd_rbf(x[rand_idx], x, bw)
    mmd_self = mmd_rbf(x, x, bw)

    # determinism probes
    thin_again = stein_thin(x, score, m, bw)
    determinism = float(np.array_equal(thin_idx, thin_again))

    # weaker-correlation panel: thinning edge should persist
    x2, s2 = synth_posterior(160, 4, 0.2, seed + 1)
    bw2 = median_bandwidth(x2)
    thin2 = stein_thin(x2, s2, m, bw2)
    rand2 = rng.choice(x2.shape[0], size=m, replace=False)
    mmd_thin2 = mmd_rbf(x2[thin2], x2, bw2)
    mmd_rand2 = mmd_rbf(x2[rand2], x2, bw2)

    rep = compress_report(x, m, bw, seed)

    return {
        "synthetic_thin_mmd_edge": mmd_thin / mmd_rand if mmd_rand > 0 else 1.0,
        "synthetic_thin_mmd_edge_weakcorr": (mmd_thin2 / mmd_rand2 if mmd_rand2 > 0 else 1.0),
        "synthetic_herd_edge": mmd_herd / mmd_rand if mmd_rand > 0 else 1.0,
        "synthetic_ksd_thin_vs_random": (
            ksd(thin_idx) / ksd(rand_idx) if ksd(rand_idx) > 0 else 1.0
        ),
        "synthetic_ksd_thin_vs_full": ksd(thin_idx) / ksd(np.arange(x.shape[0])),
        "synthetic_mmd_self": mmd_self,
        "synthetic_mmd_herd": mmd_herd,
        "synthetic_report_edge": rep["mmd_edge"],
        "synthetic_overlap_thin_herd": float(np.intersect1d(thin_idx, herd_idx).size) / m,
        "synthetic_determinism": determinism,
    }
