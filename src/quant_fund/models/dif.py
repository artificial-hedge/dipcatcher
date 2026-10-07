"""Differential item functioning — Mantel-Haenszel and logistic- (SYNTHETIC)
regression DIF screens.

Holland & Thayer (1988): stratifying a binary item's 2x2 table
by total-score strata, the MH common-odds estimate

    Delta_MH = -2.35 log( alpha_MH ),  alpha_MH =
        sum_s A_s B_s / N_s  /  sum_s C_s D_s / N_s
        (A,D focal/ref correct-by-rest cells)

classifies items by ETS A/B/C rules (|Delta| < 1 = A; 1-1.5 = B;
> 1.5 = C conditional on significance). The MH chi^2 tests
conditional independence. Swaminathan & Rogers (1990) logistic-
regression DIF compares nested models (score vs score+group vs
+interaction) via deviance tests — uniform vs nonuniform DIF.

Honesty: the bench plants a DIF item (focal-group success
deflated at fixed score) and a null item; strata use total-score
quintiles as is standard with small n. Fail-closed on empty
strata, non-binary items, or degenerate tables.

References: Holland & Thayer (1988) "Differential item
performance and the Mantel-Haenszel procedure"; Swaminathan &
Rogers (1990) "Detecting differential item functioning using
logistic regression"; Zwick (1990) ETS A/B/C classification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def _check(
    item: FloatArray, score: FloatArray, group: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(item, dtype=float)
    s = np.asarray(score, dtype=float)
    g = np.asarray(group, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or g.shape != y.shape or y.size < 30:
        raise ValueError("bad inputs")
    if not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("non-finite input")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("non-binary item")
    if set(np.unique(g)) - {0.0, 1.0}:
        raise ValueError("group must be 0/1")
    return y, s, g


def _stratify(s: FloatArray, n_strata: int = 5) -> FloatArray:
    qs = np.quantile(s, np.linspace(0, 1, n_strata + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    return np.digitize(s, qs[1:-1]).astype(np.float64)


def mh_dif(
    item: FloatArray, score: FloatArray, group: FloatArray, n_strata: int = 5
) -> dict[str, float]:
    """Mantel-Haenszel DIF: alpha_MH, Delta_MH (ETS scale), chi^2 p."""
    y, s, g = _check(item, score, group)
    strat = _stratify(s, n_strata)
    num = 0.0
    den = 0.0
    stat_num = 0.0
    stat_var = 0.0
    for sv in np.unique(strat):
        m = strat == sv
        yv, gv = y[m], g[m]
        ns = m.sum()
        a = ((gv == 1) & (yv == 1)).sum()  # focal correct
        b = ((gv == 1) & (yv == 0)).sum()  # focal wrong
        c = ((gv == 0) & (yv == 1)).sum()  # ref correct
        d = ((gv == 0) & (yv == 0)).sum()  # ref wrong
        n1 = a + b  # focal n
        n0 = c + d  # ref n
        k1 = a + c  # correct n
        if ns <= 1 or n1 == 0 or n0 == 0 or k1 == 0 or (ns - k1) == 0:
            continue
        num += a * d / ns
        den += b * c / ns
        # MH chi^2 contribution (Cochran-Mantel-Haenszel form)
        e_a = n1 * k1 / ns
        v_a = n1 * n0 * k1 * (ns - k1) / (ns * ns * (ns - 1))
        stat_num += a - e_a
        stat_var += v_a
    if den <= 0 or stat_var <= 0:
        raise ValueError("degenerate MH table")
    alpha_mh = num / den
    delta = -2.35 * np.log(alpha_mh)
    cmh = (abs(stat_num) - 0.5) ** 2 / stat_var
    p = float(chi2.sf(cmh, 1))
    # ETS A/B/C classification
    if abs(delta) < 1.0 or p > 0.05:
        cls = 0.0  # A
    elif abs(delta) < 1.5:
        cls = 1.0  # B
    else:
        cls = 2.0  # C
    return {
        "alpha_mh": float(alpha_mh),
        "delta_mh": float(delta),
        "chi2": float(cmh),
        "p": p,
        "ets_class": cls,
    }


def logistic_dif(item: FloatArray, score: FloatArray, group: FloatArray) -> dict[str, float]:
    """Swaminathan-Rogers logistic DIF via nested deviance tests.

    Fits three models by Newton IRLS: (i) y ~ s, (ii) + g,
    (iii) + s*g. Uniform DIF = LRT(ii vs i); nonuniform = LRT
    (iii vs ii).
    """
    y, s, g = _check(item, score, group)
    n = y.size
    sc = (s - s.mean()) / max(s.std(), 1e-9)

    def _fit(x: FloatArray) -> float:
        b = np.zeros(x.shape[1])
        for _ in range(50):
            eta = np.clip(x @ b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            w = np.clip(p * (1 - p), 1e-6, None)
            grad = x.T @ (y - p)
            h = (x.T * w) @ x
            step = np.linalg.solve(h + 1e-8 * np.eye(x.shape[1]), grad)
            b = b + step
            if np.abs(step).max() < 1e-8:
                break
        p = np.clip(1.0 / (1.0 + np.exp(-x @ b)), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    x1 = np.stack([np.ones(n), sc], axis=1)
    x2 = np.stack([np.ones(n), sc, g], axis=1)
    x3 = np.stack([np.ones(n), sc, g, sc * g], axis=1)
    d1 = _fit(x1)
    d2 = _fit(x2)
    d3 = _fit(x3)
    lrt_uniform = max(0.0, 2.0 * (d1 - d2))
    lrt_nonuniform = max(0.0, 2.0 * (d2 - d3))
    return {
        "p_uniform": float(chi2.sf(lrt_uniform, 1)),
        "p_nonuniform": float(chi2.sf(lrt_nonuniform, 1)),
        "lrt_uniform": lrt_uniform,
        "lrt_nonuniform": lrt_nonuniform,
    }


def bench_dif(seed: int = 20261231 + 440) -> dict[str, float]:
    """SYNTHETIC check — planted DIF flagged, null clean."""
    rng = np.random.default_rng(seed)
    n = 600
    score = rng.standard_normal(n)
    group = (rng.random(n) < 0.5).astype(float)
    # DIF item: focal group gets -1.5 logit penalty
    logit = 1.2 * score - 1.5 * group
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)
    out = mh_dif(y, score, group)
    lg = logistic_dif(y, score, group)
    # null item: no group effect
    p0 = 1.0 / (1.0 + np.exp(-1.2 * score))
    y0 = (rng.random(n) < p0).astype(float)
    out0 = mh_dif(y0, score, group)
    if out["p"] > 0.05 or out["delta_mh"] <= 1.0 or out0["p"] < 0.01 or lg["p_uniform"] > 0.05:
        raise ValueError(
            f"dif off: delta={out['delta_mh']:.3f} p={out['p']:.4f} "
            f"null={out0['p']:.4f} lg={lg['p_uniform']:.4f}"
        )
    return {
        "synthetic_dif_delta": out["delta_mh"],
        "synthetic_dif_p": out["p"],
        "synthetic_dif_null_p": out0["p"],
        "synthetic_dif_logistic_p": lg["p_uniform"],
        "synthetic_score": 1.0,
    }
