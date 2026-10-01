"""Intraclass correlation coefficients — McGraw & Wong (1996).

For an (n targets, k raters) matrix the one-way/two-way ANOVA
mean squares give the standard ICC family:

    ICC(1,1) = (MS_B - MS_W) / (MS_B + (k-1) MS_W)          one-way
    ICC(2,1) = (MS_B - MS_E) / (MS_B + (k-1) MS_E + k(MS_J - MS_E)/n)
    ICC(3,1) = (MS_B - MS_E) / (MS_B + (k-1) MS_E)          consistency
    ICC(A,1) = ICC(2,1)   (absolute agreement, random raters)
    ICC(C,1) = ICC(3,1)   (consistency)

with MS_B between-targets, MS_J between-raters, MS_E residual.
Shrout & Fleiss (1979) forms (1,1)/(2,1)/(3,1) map one-to-one.
The standard error of measurement SEM = sd * sqrt(1 - ICC).

Honesty: the bench uses planted rater agreement (ICC high) and
independent raters (ICC ~ 0 or slightly negative, floor-reported).
Negative ICC is a real outcome (MS_B < MS_E) — reported, not
clipped. Fail-closed on single-rater or degenerate designs.

References: Shrout & Fleiss (1979) "Intraclass correlations:
uses in assessing rater reliability"; McGraw & Wong (1996)
"Forming inferences about some intraclass correlation
coefficients"; Koo & Li (2016) guideline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] < 2:
        raise ValueError("bad rater matrix")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    return a


def icc(x: FloatArray) -> dict[str, float]:
    """All Shrout-Fleiss ICC forms on an (n, k) matrix."""
    a = _check(x)
    n, k = a.shape
    grand = a.mean()
    row_mean = a.mean(axis=1)
    col_mean = a.mean(axis=0)
    ss_b = k * float(((row_mean - grand) ** 2).sum())
    ss_j = n * float(((col_mean - grand) ** 2).sum())
    ss_t = float(((a - grand) ** 2).sum())
    ss_e = ss_t - ss_b - ss_j
    df_b, df_j, df_e = n - 1, k - 1, (n - 1) * (k - 1)
    ms_b = ss_b / df_b
    ms_j = ss_j / df_j
    ms_e = ss_e / max(df_e, 1)
    ms_w = (ss_j + ss_e) / (n * (k - 1))
    icc_11 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if ms_b + (k - 1) * ms_w > 0 else 0.0
    denom_2 = ms_b + (k - 1) * ms_e + k * (ms_j - ms_e) / n
    icc_21 = (ms_b - ms_e) / denom_2 if denom_2 > 0 else 0.0
    denom_3 = ms_b + (k - 1) * ms_e
    icc_31 = (ms_b - ms_e) / denom_3 if denom_3 > 0 else 0.0
    sd = float(a.std(ddof=1))
    return {
        "icc_1_1": float(icc_11),
        "icc_2_1": float(icc_21),
        "icc_3_1": float(icc_31),
        "icc_a_1": float(icc_21),
        "icc_c_1": float(icc_31),
        "sem": sd * np.sqrt(max(1.0 - icc_21, 0.0)),
        "ms_b": float(ms_b),
        "ms_e": float(ms_e),
    }


def bench_icc(seed: int = 20261231 + 439) -> dict[str, float]:
    """SYNTHETIC check — shared signal high ICC, noise low."""
    rng = np.random.default_rng(seed)
    n, k = 60, 4
    truth = rng.standard_normal(n) * 2.0
    agree = truth[:, None] + 0.4 * rng.standard_normal((n, k))
    out = icc(agree)
    noise = rng.standard_normal((n, k))
    out_n = icc(noise)
    if out["icc_2_1"] < 0.8 or out_n["icc_2_1"] > 0.4:
        raise ValueError(f"icc off: agree={out['icc_2_1']:.3f} noise={out_n['icc_2_1']:.3f}")
    return {
        "synthetic_icc_agree": out["icc_2_1"],
        "synthetic_icc_noise": out_n["icc_2_1"],
        "synthetic_icc_sem": out["sem"],
        "score": 1.0,
    }
