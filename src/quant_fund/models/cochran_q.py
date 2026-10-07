"""Cochran's Q — k related binary samples.

Cochran (1950): for an (n blocks, k treatments) binary matrix the
test of equal column proportions is

    Q = k (k-1) sum_j (T_j - Tbar)^2 / sum_i L_i (k - L_i)
      ~ chi^2_{k-1}

where T_j are column totals and L_i row totals. Blocks with
L_i in {0, k} contribute nothing and drop out of the denominator.

Honesty: the bench uses a planted differential response probability
per column (rejected) and iid columns (size held). Rows with zero
or k successes carry no information (standard). Fail-closed on
non-binary input, degenerate denominators, or fewer than 3 columns
(use McNemar for k=2).

References: Cochran (1950) "The comparison of percentages in
matched samples", Biometrika 37:256; Hollander, Wolfe, Chicken
(2014) ch. 7.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def cochran_q(x: FloatArray) -> dict[str, float]:
    """Cochran's Q on an (n, k) binary block design (k >= 3)."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or a.shape[0] < 5:
        raise ValueError("bad design")
    if not np.isfinite(a).all() or not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("non-binary input")
    n, k = a.shape
    t = a.sum(axis=0)
    l_i = a.sum(axis=1)
    denom = float((l_i * (k - l_i)).sum())
    if denom <= 0:
        raise ValueError("no informative blocks")
    q = float(k * (k - 1) * ((t - t.mean()) ** 2).sum() / denom)
    return {
        "q": q,
        "p": float(chi2.sf(q, k - 1)),
        "informative_blocks": float((l_i * (k - l_i) > 0).sum()),
    }


def bench_cochran_q(seed: int = 20261231 + 434) -> dict[str, float]:
    """SYNTHETIC check — planted column effect rejected, null held."""
    rng = np.random.default_rng(seed)
    n, k = 60, 4
    probs = np.array([0.2, 0.35, 0.55, 0.75])
    x = (rng.random((n, k)) < probs[None, :]).astype(float)
    out = cochran_q(x)
    xn = (rng.random((n, k)) < 0.5).astype(float)
    out_n = cochran_q(xn)
    if out["p"] > 0.01 or out_n["p"] < 0.005:
        raise ValueError(f"cochran_q off: p={out['p']:.4f} null={out_n['p']:.4f}")
    return {
        "synthetic_cochran_q_p": out["p"],
        "synthetic_cochran_q_p_null": out_n["p"],
        "synthetic_cochran_q_stat": out["q"],
        "synthetic_score": 1.0,
    }
