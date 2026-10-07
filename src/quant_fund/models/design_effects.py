"""Design-effect diagnostics — Kish DEFF, effective sample size, (SYNTHETIC)
and weighting-cost decomposition.

Kish (1965, 1992): unequal weights inflate the variance of a
mean by

    deff_w = 1 + CV_w^2 = n sum w_i^2 / (sum w_i)^2

so the effective sample size is n_eff = n / deff. The full design
effect of a clustered weighted design decomposes as
    deff ~ deff_weights * deff_clustering
(Gabler, Häder, Lahiri 1999 model-based form). Weight-trimming
trade-offs are summarized by the DEFF curve at common caps.

Honesty: DEFF is a variance-ratio summary, not a lossless sample
size — reported alongside the CV of weights and the share of
weight in the top decile (concentration diagnostic). The bench
uses mildly and severely variable weights where n_eff ordering
and the deff formula are checkable in closed form. Fail-closed
on non-positive weights.

References: Kish (1965) "Survey Sampling" sect. 11.7; Kish (1992)
"Weighting for unequal P_i", J. Off. Stat. 8:183; Gabler, Häder,
Lahiri (1999) "A model based justification of Kish's formula";
Potter (1990) DEFF studies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_w(weights: FloatArray) -> FloatArray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size < 4 or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("bad weights")
    return w


def design_effect(weights: FloatArray) -> dict[str, float]:
    """Kish DEFF + n_eff + weight-concentration diagnostics."""
    w = _check_w(weights)
    n = w.size
    cv2 = float((w.std() / w.mean()) ** 2)
    deff = 1.0 + cv2
    n_eff = n / deff
    order = np.sort(w)
    top_share = float(order[-max(1, n // 10) :].sum() / w.sum())
    return {
        "deff": deff,
        "n_eff": float(n_eff),
        "cv2_weights": cv2,
        "top_decile_share": top_share,
        "w_max_over_mean": float(w.max() / w.mean()),
    }


def trimmed_deff_curve(
    weights: FloatArray, quantiles: tuple[float, ...] = (0.9, 0.95, 0.99)
) -> FloatArray:
    """DEFF if weights were Winsorized at each upper quantile.

    Returns parallel array of DEFF values (lower = less variance
    inflation but more bias risk — the standard trade-off).
    """
    w = _check_w(weights)
    out = np.empty(len(quantiles))
    for i, q in enumerate(quantiles):
        cap = np.quantile(w, q)
        wt = np.minimum(w, cap)
        cv2 = float((wt.std() / wt.mean()) ** 2)
        out[i] = 1.0 + cv2
    return np.asarray(out, dtype=np.float64)


def bench_design_effects(seed: int = 20261231 + 449) -> dict[str, float]:
    """SYNTHETIC check — DEFF formula verified, trimming reduces it."""
    rng = np.random.default_rng(seed)
    n = 500
    w_mild = np.exp(0.3 * rng.standard_normal(n))
    w_bad = np.exp(1.1 * rng.standard_normal(n))
    d_mild = design_effect(w_mild)
    d_bad = design_effect(w_bad)
    # closed form: deff = n * sum w^2 / (sum w)^2
    exact = n * float((w_bad * w_bad).sum()) / float(w_bad.sum() ** 2)
    curve = trimmed_deff_curve(w_bad)
    if (
        abs(d_bad["deff"] - exact) > 1e-9
        or d_bad["deff"] <= d_mild["deff"]
        or curve[0] >= d_bad["deff"]
    ):
        raise ValueError(
            f"deff off: bad={d_bad['deff']:.3f} exact={exact:.3f} "
            f"mild={d_mild['deff']:.3f} trim={curve[0]:.3f}"
        )
    return {
        "synthetic_deff_mild": d_mild["deff"],
        "synthetic_deff_bad": d_bad["deff"],
        "synthetic_deff_exact": exact,
        "synthetic_deff_trimmed": float(curve[0]),
        "synthetic_score": 1.0,
    }
