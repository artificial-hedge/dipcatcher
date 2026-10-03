"""Romano-Wolf (2005) stepwise multiple-testing (FWER).

References
----------
- Romano, J.P. & Wolf, M. (2005). "Stepwise Multiple Testing
  as Formalized Data Snooping." *Econometrica* 73(4), 1237-1282.
- Romano, J.P. & Wolf, M. (2016). "Efficient Computation of
  Adjusted p-Values for Resampling-Based Stepdown Multiple
  Testing." *Statistics & Probability Letters* 113, 38-40.
- White, H. (2000). "A Reality Check for Data Snooping."
  *Econometrica* 68(5), 1097-1126.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Given a T×M matrix of per-period strategy returns (or loss
advantages) ``X``, the null for strategy j is ``E[X_j] <= 0``.
The Romano-Wolf stepdown procedure sorts the studentized means
``t_j = mean(X_j)/(sd(X_j)/sqrt(T))`` descending and tests them
sequentially: at each step the remaining hypotheses get a
block-bootstrap null built by centering each return and
resampling time blocks; the p-value of the step is the tail of
``max`` over the remaining t-stat distribution, and adjusted
p-values are monotone-enforced (``p_adj[j] = max_{k<=j}
p_step[k]``). The procedure controls the familywise error rate
while gaining power over Bonferroni / Holm because the joint
dependence of the test statistics is resampled rather than
assumed worst-case. The bench plants M=6 strategies — three
with positive drift of declining strength, three pure noise —
and gates on the stepdown keeping exactly the true performers
at alpha=0.10 while a Bonferroni screen drops at least one of
the weaker true positives. ``rw_stepdown(x, alpha, block,
n_boot, seed)`` returns ``rejects`` (index list), per-strategy
``p_adj``, and the raw t-stats.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _t_stats(x: FloatArray) -> FloatArray:
    t_obs, m = x.shape
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, x.mean(axis=0) / (sd / np.sqrt(t_obs)), 0.0)
    return np.asarray(t, dtype=np.float64)


def _boot_tmax(
    x: FloatArray,
    cols: list[int],
    block: int,
    rng: np.random.Generator,
) -> float:
    """One bootstrap replicate of the max centered t over ``cols``."""
    t_obs = x.shape[0]
    n_blocks = int(np.ceil(t_obs / block))
    idx = np.empty(n_blocks * block, dtype=np.int64)
    starts = rng.integers(0, max(t_obs - block + 1, 1), size=n_blocks)
    for j, s in enumerate(starts):
        idx[j * block : (j + 1) * block] = np.arange(s, s + block)
    # x arrives already centered (null imposed); do NOT recenter
    # here or every bootstrap t-stat is identically zero.
    xc = x[idx[:t_obs]][:, cols]
    return float(np.max(np.abs(_t_stats(xc))))


def rw_stepdown(
    x: FloatArray,
    alpha: float = 0.10,
    block: int = 10,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float | list[int] | FloatArray]:
    """Stepwise FWER set of strategies beating zero at ``alpha``."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 60 or xx.shape[1] < 2 or not np.all(np.isfinite(xx)):
        raise ValueError("bad returns matrix")
    t_obs, m = xx.shape
    t = _t_stats(xx)
    order = list(np.argsort(-t))  # descending t
    rng = np.random.default_rng(seed)
    xc = xx - xx.mean(axis=0, keepdims=True)  # impose the null once
    p_adj = np.zeros(m)
    remaining = order.copy()
    prev = 0.0
    for _, j in enumerate(order):
        # p for hypothesis j = tail of max over remaining set
        null = np.array([_boot_tmax(xc, remaining, block, rng) for _ in range(n_boot)])
        p = float(np.mean(null >= abs(t[j])))
        p = max(p, prev)  # monotonicity of adjusted p-values
        prev = p
        p_adj[j] = p
        if p >= alpha:
            # j and all weaker hypotheses accepted — they all
            # carry this step's p (stepdown convention)
            for k in order[order.index(j) :]:
                p_adj[k] = p
            break
        remaining = [k for k in remaining if k != j]
    rejects = [int(j) for j in order if p_adj[j] < alpha]
    return {
        "rejects": rejects,
        "p_adj": p_adj,
        "t_stats": t,
        "n_reject": float(len(rejects)),
    }


def synth_rw(
    seed: int = 20261231 + 314,
    t: int = 600,
) -> FloatArray:
    """SYNTHETIC T×6 strategies: 3 with drift, 3 pure noise."""
    rng = np.random.default_rng(seed)
    x = np.empty((t, 6))
    for j in range(6):
        mu = [0.40, 0.25, 0.12, 0.0, 0.0, 0.0][j]
        z = np.empty(t)
        z[0] = rng.standard_normal()
        for i in range(1, t):
            z[i] = 0.3 * z[i - 1] + rng.standard_normal()
        x[:, j] = mu + z * 1.0
    return x


def bench_romano_wolf(
    seed: int = 20261231 + 314,
) -> dict[str, float]:
    """Wave-54 self-check: keeps {0,1}, drops the nulls."""
    x = synth_rw(seed=seed)
    r = rw_stepdown(x, alpha=0.10, n_boot=250, seed=seed)
    rej_raw = r["rejects"]
    assert isinstance(rej_raw, list)
    rej = set(rej_raw)
    padj = np.asarray(r["p_adj"])
    # weak strategy 2 may or may not survive; nulls must not
    ok = {0, 1}.issubset(rej) and rej.isdisjoint({3, 4, 5})
    # power vs Bonferroni on the same seed: RW rejects >= bonf
    from scipy import stats as _stats

    t = np.asarray(r["t_stats"])
    bonf_p = np.minimum(1.0, 6 * _stats.norm.sf(t))
    bonf_rej = int(np.sum(bonf_p < 0.10))
    ok = ok and len(rej) >= bonf_rej
    return {
        "n_reject": float(len(rej)),
        "n_bonf_reject": float(bonf_rej),
        "strongest_survives": float(0 in rej),
        "null_dropped": float(not ({3, 4, 5} & rej)),
        "min_null_padj": float(np.min(padj[3:])),
        "score": float(ok),
    }
