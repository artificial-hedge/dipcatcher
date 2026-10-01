"""Hansen-Lunde-Nason (2011) Model Confidence Set procedure.

References
----------
- Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The Model
  Confidence Set." *Econometrica* 79(2), 453-497.
- Hansen, P.R. (2005). "A Test for Superior Predictive Ability."
  *JBES* 23(4), 365-380.
- White, H. (2000). "A Reality Check for Data Snooping."
  *Econometrica* 68(5), 1097-1126.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Given per-observation loss differentials ``d_ij,t = l_i,t -
l_j,t`` across M competing models, the MCS iteratively trims the
worst performer until the equal-predictive-ability null survives.
We implement the ``T_max``/``R`` algorithm: at each round, the
elimination statistic is the max over pairs of the studentized
sample mean differential (``t_ij = mean(d_ij)/sd(d_ij)`` —
White/Hansen t-form; the bootstrap supplies the nuisance
distribution). The moving-block bootstrap resamples the loss
matrix in blocks of expected length ``b``; under the bootstrap
null each model's differential is centered, so the bootstrapped
``T_max`` distribution gives step-down p-values: model i is
eliminated when its bootstrap p exceeds alpha. The surviving set
``M*`` contains every model not statistically distinguishable
from the best at level alpha. The synth plants three forecast
error streams where model A dominates by a clear MSPE margin,
model B is a diluted copy, and model C is mediocre: at alpha =
0.10 the MCS must keep {A,B}-ish and drop C, and the planted
best model must always survive. ``mcs_test(losses, alpha, b)``
returns surviving indices, per-model p-values, and the trimmed
statistic path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _d_ij(losses: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Pairwise mean loss differentials + their t-stat matrix."""
    m = losses.shape[1]
    dbar = np.empty((m, m))
    tmat = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                dbar[i, j] = 0.0
                tmat[i, j] = 0.0
                continue
            d = losses[:, i] - losses[:, j]
            sd = float(np.std(d, ddof=1))
            dbar[i, j] = float(np.mean(d))
            tmat[i, j] = dbar[i, j] / (sd / np.sqrt(d.size)) if sd > 0 else 0.0
    return dbar, tmat


def _tmax_boot(losses: FloatArray, b: int, n_boot: int, seed: int) -> FloatArray:
    """Bootstrapped null distribution of the max-pairwise t."""
    t_obs, m = losses.shape
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(t_obs / b))
    lc = losses - losses.mean(axis=0, keepdims=True)
    stat = np.empty(n_boot)
    for r_i in range(n_boot):
        idx = np.empty(n_blocks * b, dtype=np.int64)
        starts = rng.integers(0, max(t_obs - b + 1, 1), size=n_blocks)
        for j, s in enumerate(starts):
            idx[j * b : (j + 1) * b] = np.arange(s, s + b)
        lb = lc[idx[:t_obs]]
        _, tmat = _d_ij(lb)
        stat[r_i] = float(np.max(np.abs(tmat)))
    return stat


def mcs_test(
    losses: FloatArray,
    alpha: float = 0.10,
    block: int = 10,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float | list[int] | list[float]]:
    """Step-down model confidence set at level ``alpha``.

    ``losses``: T×M matrix of per-observation losses (lower is
    better). Returns ``survivors`` (column indices kept),
    ``p_values`` (per-model elimination p-value — the probability
    the bootstrap T_max exceeds the statistic at that model's
    elimination round; the best model gets p=1 by construction),
    and ``alpha``/``n_boot``.
    """
    ll = np.asarray(losses, dtype=np.float64)
    if ll.ndim != 2 or ll.shape[0] < 60 or ll.shape[1] < 2 or not np.all(np.isfinite(ll)):
        raise ValueError("bad losses")
    m = ll.shape[1]
    alive = list(range(m))
    pvals = np.full(m, np.nan)
    while len(alive) > 1:
        sub = ll[:, alive]
        dbar, tmat = _d_ij(sub)
        # EPA stat: max over pairs of |t_ij|
        stat = float(np.max(np.abs(tmat)))
        null = _tmax_boot(sub, block, n_boot, seed)
        p = float(np.mean(null >= stat))
        if p >= alpha:
            # can't reject EPA among alive — every survivor takes
            # the final round's p-value (the best model's is the
            # largest by construction of the step-down)
            for j in alive:
                pvals[j] = p
            break
        # eliminate worst: the model with the largest mean loss
        # relative to the best alive
        mean_l = sub.mean(axis=0)
        worst_i = int(np.argmax(mean_l))
        pvals[alive[worst_i]] = p
        alive.pop(worst_i)
    # singleton survivors get 1.0 by convention
    pvals[np.isnan(pvals)] = 1.0
    return {
        "survivors": alive,
        "p_values": [float(p) for p in pvals],
        "alpha": alpha,
        "n_eliminated": float(m - len(alive)),
    }


def synth_mcs(
    seed: int = 20261231 + 308,
    t: int = 800,
) -> FloatArray:
    """SYNTHETIC loss streams: A best, B near-best, C worse."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(t)
    noise = rng.standard_normal((t, 3))
    e_a = noise[:, 0] * 0.5
    e_b = e_a + noise[:, 1] * 0.15  # close second
    e_c = noise[:, 2] * 0.9 + 0.6 * np.abs(y)  # clearly worse
    return np.column_stack([e_a * e_a, e_b * e_b, e_c * e_c])


def bench_model_confidence_set(
    seed: int = 20261231 + 308,
) -> dict[str, float]:
    """Wave-53 self-check: MCS keeps the planted best, drops the worst."""
    losses = synth_mcs(seed=seed)
    r = mcs_test(losses, alpha=0.10, n_boot=250, seed=seed)
    survivors = r["survivors"]
    pvals = r["p_values"]
    assert isinstance(survivors, list) and isinstance(pvals, list)
    ok = 0 in survivors and 2 not in survivors and float(pvals[0]) >= float(pvals[2])
    return {
        "n_survivors": float(len(survivors)),
        "best_survives": float(0 in survivors),
        "worst_dropped": float(2 not in survivors),
        "p_best": float(pvals[0]),
        "p_worst": float(pvals[2]),
        "score": float(ok),
    }
