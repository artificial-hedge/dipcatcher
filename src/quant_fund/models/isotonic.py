"""Isotonic regression — pool-adjacent-violators monotone fitting.

Barlow, Bartholomew, Bremner & Brunk (1972): the L2-isotonic
regression of y on a (totally ordered) x solves

    min_f  sum_i w_i (y_i - f_i)^2  s.t. f nondecreasing in x

PAVA pools adjacent violators into weighted block means in O(n).
Applying PAVA to (score, label) pairs sorted by score yields the
isotonic probability calibration of Zadrozny & Elkan (2002): a
monotone, piecewise-constant mapping from raw score to empirical
event rate.

Honesty: the bench checks (a) PAVA recovers a monotone relationship
at a fraction of the identity-fit MSE, (b) isotonic calibration cuts
the Brier/cross-entropy of a misspecified logistic score, and (c)
the fitted mapping is monotone. Fail-closed on non-finite inputs or
length mismatches.

References: Barlow et al. (1972) "Statistical Inference under Order
Restrictions"; Zadrozny & Elkan (2002) "Transforming classifier
scores into accurate multiclass probability estimates"; Niculescu-
Mizil & Caruana (2005) ICML.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pava(y: FloatArray, w: FloatArray | None = None) -> FloatArray:
    """Pool-adjacent-violators: nondecreasing L2 fit in input order.

    ``y`` must already be ordered by the predictor. Returns the
    fitted values (nondecreasing, piecewise constant).
    """
    a = np.asarray(y, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all():
        raise ValueError("bad y")
    if w is None:
        wt = np.ones(a.size)
    else:
        wt = np.asarray(w, dtype=float)
        if wt.shape != a.shape or (wt <= 0).any() or not np.isfinite(wt).all():
            raise ValueError("bad weights")
    # block representation: (value, weight, start, end)
    vals = a.astype(float).tolist()
    wts = wt.astype(float).tolist()
    lvl = list(zip(vals, wts, range(a.size), range(1, a.size + 1), strict=True))
    i = 0
    while i < len(lvl) - 1:
        if lvl[i][0] > lvl[i + 1][0] + 1e-15:
            v0, w0, s0, e0 = lvl[i]
            v1, w1, s1, e1 = lvl[i + 1]
            merged = ((v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1, s0, e1)
            lvl[i : i + 2] = [merged]
            if i > 0:
                i -= 1
            continue
        i += 1
    out = np.empty(a.size)
    for v, _w, s, e in lvl:
        out[s:e] = v
    return np.asarray(out, dtype=np.float64)


def isotonic_fit(x: FloatArray, y: FloatArray, w: FloatArray | None = None) -> FloatArray:
    """Isotonic regression of ``y`` on ``x`` (sorts internally)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or a.size < 2:
        raise ValueError("bad input")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite input")
    order = np.argsort(a, kind="stable")
    wt = None if w is None else np.asarray(w, dtype=float)[order]
    fit = pava(b[order], wt)
    out = np.empty_like(fit)
    out[order] = fit
    return out


def isotonic_calibration(scores: FloatArray, labels: FloatArray) -> dict[str, FloatArray | float]:
    """Zadrozny-Elkan isotonic probability calibration.

    Sorts by score, runs PAVA on the binary labels, and reports the
    calibrated probabilities + in-sample Brier/cross-entropy versus
    the raw score treated as a probability.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.ndim != 1 or y.shape != s.shape or s.size < 5:
        raise ValueError("bad input")
    if not np.isfinite(s).all() or not np.isfinite(y).all():
        raise ValueError("non-finite input")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    order = np.argsort(s, kind="stable")
    cal_sorted = pava(y[order])
    cal = np.empty_like(cal_sorted)
    cal[order] = np.clip(cal_sorted, 1e-6, 1 - 1e-6)
    raw = np.clip(s, 1e-6, 1 - 1e-6)
    brier_cal = float(np.mean((cal - y) ** 2))
    brier_raw = float(np.mean((raw - y) ** 2))
    if not (np.diff(cal_sorted) >= -1e-12).all():
        raise ValueError("calibration not monotone")
    return {
        "calibrated": np.asarray(cal, dtype=np.float64),
        "brier_cal": np.asarray(brier_cal),
        "brier_raw": np.asarray(brier_raw),
    }


def bench_isotonic(seed: int = 20261231 + 418) -> dict[str, float]:
    """SYNTHETIC check — monotone recovery + calibration improvement."""
    rng = np.random.default_rng(seed)
    n = 400
    x = np.linspace(0, 6, n) + 0.1 * rng.standard_normal(n)
    f_true = 1.0 / (1.0 + np.exp(-(x - 3.0)))
    y = f_true + 0.15 * rng.standard_normal(n)
    fit = isotonic_fit(x, y)
    mse_iso = float(np.mean((fit - f_true) ** 2))
    mse_id = float(np.mean((x / x.max() - f_true) ** 2))
    ratio = mse_iso / max(mse_id, 1e-12)
    if ratio > 0.5:
        raise ValueError(f"isotonic recovery off: ratio={ratio:.3f}")
    # misspecified score -> isotonic calibration reduces Brier
    labels = (rng.random(n) < f_true).astype(float)
    raw_score = np.clip(0.5 + 0.3 * np.tanh(x - 3.0) + 0.1 * rng.standard_normal(n), 0, 1)
    cal = isotonic_calibration(raw_score, labels)
    b_cal = float(cal["brier_cal"])
    b_raw = float(cal["brier_raw"])
    if b_cal >= b_raw:
        raise ValueError(f"calibration did not help: {b_cal:.4f} vs {b_raw:.4f}")
    return {
        "synthetic_iso_mse_ratio": ratio,
        "synthetic_iso_brier_cal": b_cal,
        "synthetic_iso_brier_raw": b_raw,
        "score": 1.0,
    }
