"""Distance covariance — nonparametric dependence of any form.

dCov(X,Y) = 0 iff X ⊥ Y — it catches monotone, non-monotone and
oscillatory dependence where Pearson/Spearman report ~0. Computed
from doubly-centered pairwise distance matrices:

  dCov² = mean_ij a_ij·b_ij,  a_ij = |x_i-x_j| - rowmean_i - colmean_j + grandmean

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure detection power on generated
nonlinear dependence — never market evidence.

References:
- Székely, G. J., Rizzo, M. L., Bakirov, N. K. (2007). Measuring
  and testing dependence by correlation of distances. *Annals of
  Statistics* 35, 2769-2794 — dCov/dCor and the consistency
  theorem (dCor = 0 ⟺ independence).
- Székely, G. J., Rizzo, M. L. (2009). Brownian distance
  covariance. *Annals of Applied Statistics* 3 — the asymptotic
  test whose permutation version is implemented here.
- Edelmann, D. B., Richards, D., Vogel, D. (2020). The distance
  standard deviation. *Computational Statistics* 35.

Composition: pure numpy — O(n²) distance matrices, permutation
p-value; deterministic ``np.random.default_rng``; no new deps.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = xx.size
    if n != yy.size or n < 40:
        raise ValueError("x and y must share length >= 40")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("finite inputs required")
    if np.ptp(xx) < 1e-12 or np.ptp(yy) < 1e-12:
        raise ValueError("both variables must vary")
    return xx, yy


def _distmat(v: FloatArray) -> FloatArray:
    diff = v[:, None] - v[None, :]
    return np.abs(diff)


def distance_covariance(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """dCov², dCor and Pearson |r| for contrast."""
    xx, yy = _check(x, y)
    a = _distmat(xx)
    b = _distmat(yy)
    # double-center
    a = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    b = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(a * b))
    dvx = float(np.mean(a * a))
    dvy = float(np.mean(b * b))
    dcor = math.sqrt(max(dcov2, 0.0) / math.sqrt(max(dvx * dvy, 1e-300)))
    pearson = float(abs(np.corrcoef(xx, yy)[0, 1]))
    return {
        "n": float(xx.size),
        "dcov2": dcov2,
        "dcor": dcor,
        "pearson_abs": pearson,
    }


def dcor_pvalue(x: FloatArray, y: FloatArray, n_perm: int = 120, seed: int = 0) -> float:
    """Permutation p-value for dCor(X,Y) = 0."""
    xx, yy = _check(x, y)
    obs = distance_covariance(xx, yy)["dcor"]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(xx.size)
        try:
            v = distance_covariance(xx, yy[perm])["dcor"]
        except ValueError:
            continue
        cnt += v >= obs
    return float((1 + cnt) / (1 + n_perm))


def synth_nonlinear(
    n: int = 400,
    kind: str = "circle",
    noise: float = 0.08,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Dependence DGPs where Pearson misses the link."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y: FloatArray
    if kind == "circle":
        y = np.sqrt(np.clip(1 - x**2, 0, None)) * rng.choice([-1.0, 1.0], n)
    elif kind == "sine":
        y = np.sin(4 * math.pi * x)
    elif kind == "quadratic":
        y = 4 * x**2 - 1
    elif kind == "independent":
        y = rng.uniform(-1, 1, n)
    else:
        raise ValueError(f"unknown kind {kind}")
    y = np.asarray(y, dtype=np.float64) + rng.normal(0.0, noise, n)
    return {"x": x, "y": y}


def bench_distance_covariance(
    seed: int = 20261231 + 225,
) -> dict[str, float]:
    """dCov self-check: detects nonlinear (circle) dependence that
    Pearson reports as zero; permutation test rejects.
    All ``synthetic_*``."""
    dep = synth_nonlinear(kind="quadratic", noise=0.05, seed=seed)
    d = distance_covariance(np.asarray(dep["x"]), np.asarray(dep["y"]))
    ind = synth_nonlinear(kind="independent", seed=seed + 1)
    d0 = distance_covariance(np.asarray(ind["x"]), np.asarray(ind["y"]))
    p_val = dcor_pvalue(np.asarray(dep["x"]), np.asarray(dep["y"]), seed=seed)
    p0 = dcor_pvalue(np.asarray(ind["x"]), np.asarray(ind["y"]), seed=seed + 7)
    d_b = distance_covariance(np.asarray(dep["x"]), np.asarray(dep["y"]))

    dcor = float(d["dcor"])
    return {
        "synthetic_dcor": dcor,
        "synthetic_pearson": float(d["pearson_abs"]),
        "synthetic_p_perm": float(p_val),
        "synthetic_null_dcor": float(d0["dcor"]),
        "synthetic_null_p": float(p0),
        "synthetic_detects": float(
            dcor > 0.25 and float(p_val) < 0.05 and float(d["pearson_abs"]) < 0.2
        ),
        "synthetic_determinism": float(dcor == float(d_b["dcor"])),
    }
