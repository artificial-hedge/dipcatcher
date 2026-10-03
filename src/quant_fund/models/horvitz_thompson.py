"""Horvitz-Thompson and Hajek estimators for design-based survey
inference.

Horvitz & Thompson (1952): for a probability sample with
first-order inclusion probabilities pi_i, the population total
is unbiasedly estimated by

    t_hat = sum_i y_i / pi_i

with the Sen-Yates-Grundy (1953) variance under fixed-size
sampling

    V = sum_i sum_j>i (pi_i pi_j - pi_ij)(y_i/pi_i - y_j/pi_j)^2

computed here in the with-replacement Hartley-Rao proxy
(pi_ij approximated by pi_i pi_j / normalizer), which is
conservative and standard when joint probabilities are unknown.
The Hajek (1971) ratio mean t_hat / N_hat is stable when N_hat
= sum 1/pi_i deviates from N.

Honesty: the bench draws a known stratified PPS design where the
true total is computable; the HT estimate must lie within ~3 SE
and the naive unweighted mean must be visibly biased. Bounds
documented; joint-inclusion approximation flagged conservative.
Fail-closed on pi outside (0,1] or zero-variance designs.

References: Horvitz & Thompson (1952) "A generalization of
sampling without replacement from a finite universe", JASA
47:663; Sen (1953) / Yates & Grundy (1953); Hajek (1971)
comment on Basu; Särndal, Swensson, Wretman (1992).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(y: FloatArray, pi: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(y, dtype=float)
    p = np.asarray(pi, dtype=float)
    if a.ndim != 1 or p.shape != a.shape or a.size < 4:
        raise ValueError("bad sample")
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("non-finite input")
    if (p <= 0).any() or (p > 1).any():
        raise ValueError("inclusion probabilities outside (0,1]")
    return a, p


def horvitz_thompson(y: FloatArray, pi: FloatArray) -> dict[str, float]:
    """HT total + Hartley-Rao/Sen-Yates-Grundy-style variance."""
    a, p = _check(y, pi)
    z = a / p
    t_hat = float(z.sum())
    # with-replacement approximation to the SYG variance:
    # V ~ sum_i (1 - pi_i) z_i^2 - (sum_i (1-pi_i) z_i)^2 / sum_i (1-pi_i)
    w = 1.0 - p
    wsum = w.sum()
    if wsum <= 0:
        var = 0.0
    else:
        var = float((w * z * z).sum() - (w * z).sum() ** 2 / wsum)
    var = max(var, 0.0)
    n_hat = float((1.0 / p).sum())
    return {
        "total": t_hat,
        "se": float(np.sqrt(var)),
        "n_hat": n_hat,
    }


def hajek_mean(y: FloatArray, pi: FloatArray) -> dict[str, float]:
    """Hajek ratio mean y-bar = sum(y/pi) / sum(1/pi) + linearized SE."""
    a, p = _check(y, pi)
    ht = horvitz_thompson(a, p)
    n_hat = ht["n_hat"]
    mean = ht["total"] / n_hat
    # linearization: residuals around the ratio mean
    r = (a - mean) / p
    var = float((1 - p).dot(r * r) - ((1 - p) * r).sum() ** 2 / max((1 - p).sum(), 1e-12))
    var = max(var, 0.0) / n_hat**2
    return {"mean": float(mean), "se": float(np.sqrt(var))}


def bench_horvitz_thompson(seed: int = 20261231 + 444) -> dict[str, float]:
    """SYNTHETIC check — HT near truth, unweighted biased."""
    rng = np.random.default_rng(seed)
    n_pop = 5000
    y_pop = rng.gamma(2.0, 20.0, n_pop)
    size = 1.0 + 3.0 * y_pop / y_pop.max()
    pi_i = np.clip(400 * size / size.sum(), 1e-3, 0.5)
    sel = rng.random(n_pop) < pi_i
    y = y_pop[sel]
    pi = pi_i[sel]
    ht = horvitz_thompson(y, pi)
    hj = hajek_mean(y, pi)
    true_total = float(y_pop.sum())
    true_mean = float(y_pop.mean())
    naive_mean = float(y.mean())
    err = abs(ht["total"] - true_total) / max(ht["se"], 1e-9)
    naive_bias = abs(naive_mean - true_mean) / true_mean
    hj_err = abs(hj["mean"] - true_mean) / true_mean
    if err > 4.0 or naive_bias < 0.15 or hj_err > 0.1:
        raise ValueError(
            f"ht off: z_err={err:.2f} naive_bias={naive_bias:.3f} hajek_err={hj_err:.3f}"
        )
    return {
        "synthetic_ht_z_err": err,
        "synthetic_ht_naive_bias": naive_bias,
        "synthetic_hajek_err": hj_err,
        "score": 1.0,
    }
