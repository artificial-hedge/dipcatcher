"""Data envelopment analysis (CCR/BCC frontiers) via LP.

References
----------
- Charnes, A., Cooper, W.W. & Rhodes, E. (1978). "Measuring the
  Efficiency of Decision Making Units." *European Journal of
  Operational Research* 2(6), 429-444.
- Banker, R.D., Charnes, A. & Cooper, W.W. (1984). "Some Models for
  Estimating Technical and Scale Inefficiencies in Data Envelopment
  Analysis." *Management Science* 30(9), 1078-1092.
- Simar, L. & Wilson, P.W. (1998). "Sensitivity Analysis of Efficiency
  Scores." *European Journal of Operational Research* 108(1), 75-89.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Input-oriented CCR model: for unit ``o`` solve

    min_theta theta  s.t.  sum_j lambda_j x_ij <= theta * x_io  (i in inputs)
                          sum_j lambda_j y_rj >= y_ro          (r in outputs)
                          lambda >= 0,

an LP per DMU (HiGHS). The BCC variant adds ``sum lambda = 1``.
Efficiency theta in (0,1]; slack peers identify the reference set.
The synth puts most units on a common frontier and three interior
units at known inefficiency — the CCR scores must order them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]


def dea_ccr(
    inputs: FloatArray,
    outputs: FloatArray,
    vrs: bool = False,
) -> dict[str, float | FloatArray]:
    """CCR (or BCC when ``vrs``) input-oriented efficiency scores.

    ``inputs`` (n, m) and ``outputs`` (n, s) are strictly positive
    resource/production vectors for n DMUs. Returns efficiency per
    unit plus frontier diagnostics.
    """
    xx = np.asarray(inputs, dtype=np.float64)
    yy = np.asarray(outputs, dtype=np.float64)
    if xx.ndim != 2 or yy.ndim != 2 or xx.shape[0] != yy.shape[0]:
        raise ValueError("inputs/outputs must share rows")
    n, m = xx.shape
    s = yy.shape[1]
    if n < 4 or m < 1 or s < 1:
        raise ValueError("bad dims")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("non-finite inputs")
    if np.any(xx <= 0) or np.any(yy <= 0):
        raise ValueError("inputs and outputs must be positive")

    scores = np.empty(n)
    n_ref = np.zeros(n)
    for o in range(n):
        # LP variables: [theta, lambda_1..lambda_n]
        c = np.concatenate([[1.0], np.zeros(n)])
        a_ub = np.zeros((m + s, n + 1))
        b_ub = np.zeros(m + s)
        for i in range(m):
            a_ub[i, 0] = -xx[o, i]
            a_ub[i, 1:] = xx[:, i]
            b_ub[i] = 0.0
        for r in range(s):
            a_ub[m + r, 1:] = -yy[:, r]
            b_ub[m + r] = -yy[o, r]
        if vrs:
            a_eq = np.zeros((1, n + 1))
            a_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])
        else:
            a_eq = None
            b_eq = None
        res = _opt.linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0.0, None)] + [(0.0, None)] * n,
            method="highs",
        )
        if not res.success:
            raise ValueError(f"DEA LP failed for unit {o}: {res.message}")
        scores[o] = float(res.x[0])
        lam = res.x[1:]
        n_ref[o] = float(np.count_nonzero(lam > 1e-6))

    return {
        "efficiency_mean": float(np.mean(scores)),
        "efficiency_min": float(np.min(scores)),
        "n_efficient": float(np.count_nonzero(scores > 0.999)),
        "n_referents_mean": float(np.mean(n_ref)),
        "dispersion": float(np.std(scores)),
        "_scores": scores,
    }


def synth_dea(
    n: int = 20,
    seed: int = 20261231 + 292,
    n_ineff: int = 4,
) -> dict[str, FloatArray]:
    """Mostly-frontier DMUs + a few at fixed inefficiency.

    Frontier technology y = sum(a x); the inefficient units produce
    ``1/eff`` times the frontier output.
    """
    rng = np.random.default_rng(seed)
    if n < 4:
        raise ValueError("n too small")
    xx = rng.uniform(1.0, 4.0, (n, 2))
    a_w = np.array([0.6, 0.4])
    # good units sit on the frontier (y = a'x); inefficient units are
    # scaled below it by their true score.
    yy = (xx * a_w[None, :]).sum(axis=1, keepdims=True).copy()
    idx = rng.choice(n, n_ineff, replace=False)
    eff_true = np.ones(n)
    eff_true[idx] = rng.uniform(0.4, 0.7, n_ineff)
    yy[idx] = yy[idx] * eff_true[idx][:, None]
    return {"inputs": xx, "outputs": yy, "eff_true": eff_true, "idx": idx}


def bench_dea(seed: int = 20261231 + 292) -> dict[str, float]:
    """Wave-50 self-check: inefficient units score below the frontier
    and the worst synthetic unit ranks near-last."""
    d = synth_dea(seed=seed)
    xx = np.asarray(d["inputs"])
    yy = np.asarray(d["outputs"])
    eff_true = np.asarray(d["eff_true"])
    a = dea_ccr(xx, yy)
    a2 = dea_ccr(xx, yy)
    scores = np.asarray(a["_scores"])
    worst_est = int(np.argmin(scores))
    worst_true = int(np.argmin(eff_true))
    n_eff = float(a["n_efficient"])
    eff_min = float(a["efficiency_min"])
    detects = float(
        n_eff >= 10 and scores[worst_true] <= np.sort(scores)[3] + 1e-6 and eff_min < 0.75
    )
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(np.array_equal(scores, np.asarray(a2["_scores"]))),
        "synthetic_efficiency_mean": float(a["efficiency_mean"]),
        "synthetic_efficiency_min": eff_min,
        "synthetic_n_efficient": n_eff,
        "synthetic_worst_match": float(worst_est == worst_true),
        "synthetic_dispersion": float(a["dispersion"]),
    }
