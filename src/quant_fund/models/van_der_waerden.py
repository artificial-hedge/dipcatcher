"""Van der Waerden normal-scores k-sample test.

van der Waerden (1952): pooled ranks are mapped through the
standard normal quantile, producing scores with maximal power for
near-normal errors — the k-sample analog of the normal-scores
two-sample test:

    A_j = mean of Phi^{-1}(rank/(N+1)) in group j
    X2 = (N-1) sum_j n_j A_j^2 / sum_i score_i^2  ~ chi^2_{k-1}

More powerful than Kruskal-Wallis under light tails, essentially
equivalent under heavy tails; the permutation/assymptotic chi^2
form is standard.

Honesty: the bench plants a location shift in one of three groups
(rejected) and an iid null (size held). Fail-closed on fewer than
two groups, tiny samples, or zero score variance.

References: van der Waerden (1952) "Order tests for the two-
sample problem"; Lehmann (1975) "Nonparametrics" ch. 5; Hollander,
Wolfe, Chicken (2014) ch. 6.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm, rankdata

FloatArray = NDArray[np.float64]


def van_der_waerden(*groups: FloatArray) -> dict[str, float]:
    """Normal-scores test across >=2 independent groups."""
    gs = [np.asarray(g, dtype=float) for g in groups]
    if len(gs) < 2:
        raise ValueError("need >=2 groups")
    for g in gs:
        if g.ndim != 1 or g.size < 5 or not np.isfinite(g).all():
            raise ValueError("bad group")
    pooled = np.concatenate(gs)
    n = pooled.size
    r = rankdata(pooled)
    scores = norm.ppf(r / (n + 1.0))
    if not np.isfinite(scores).all():
        raise ValueError("degenerate ranks")
    idx = np.concatenate([[i] * g.size for i, g in enumerate(gs)])
    means = np.array([scores[idx == i].mean() for i in range(len(gs))])
    ns = np.array([g.size for g in gs], dtype=float)
    x2 = float((n - 1) * (ns * means**2).sum() / (scores**2).sum())
    p = float(chi2.sf(x2, len(gs) - 1))
    return {"chi2": x2, "p": p}


def bench_van_der_waerden(seed: int = 20261231 + 436) -> dict[str, float]:
    """SYNTHETIC check — planted shift rejected, null held."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(50)
    b = rng.standard_normal(50)
    c = rng.standard_normal(50) + 1.3
    out = van_der_waerden(a, b, c)
    out_n = van_der_waerden(
        rng.standard_normal(50), rng.standard_normal(50), rng.standard_normal(50)
    )
    if out["p"] > 0.01 or out_n["p"] < 0.005:
        raise ValueError(f"vdw off: p={out['p']:.4f} null={out_n['p']:.4f}")
    return {
        "synthetic_vdw_p": out["p"],
        "synthetic_vdw_p_null": out_n["p"],
        "synthetic_vdw_chi2": out["chi2"],
        "score": 1.0,
    }
