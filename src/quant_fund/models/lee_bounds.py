"""Lee (2009) bounds on treatment effects under selection.

When the outcome is observed only conditional on a selection
variable S=1 and selection rates differ by arm, the naive
difference mixes composition and treatment effects. Lee
bounds trim the over-selected arm's outcome distribution to
match the observed-selection share, bounding the effect on
the always-selected subpopulation.

Honesty: synthetic benches apply the bounds where the true
effect is known and check bracketing — proper diagnostics,
never market evidence.

References:
- Lee, D. S. (2009). Training, wages, and sample selection:
  estimating sharp bounds on treatment effects. *Review of
  Economic Studies* 76 — the trimming bounds.
- Zhang, J. L., Rubin, D. B. (2003). Estimation of causal
  effects via principal stratification when some outcomes
  are truncated by "death". *Journal of Educational and
  Behavioral Statistics* 28 — principal strata framing.
- Imbens, G. W., Rubin, D. B. (2015). *Causal Inference for
  Statistics, Social, and Biomedical Sciences* — bounds
  discussion.
- Tauchmann, H. (2014). Lee bounds: a note on trimming.
  *Economics Letters* 123 — implementation details.

Composition: pure numpy — quantile trimming of the
over-selected arm; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lee_bounds(
    y: FloatArray,
    treat: FloatArray,
    selected: FloatArray,
) -> dict[str, float]:
    """Sharp bounds on the treatment effect for the
    always-selected stratum.

    y (observed outcome, NaN where S=0 or use only selected
    rows), treat ∈ {0,1}, selected ∈ {0,1}. Requires
    p1>P selected share in one arm; returns bound_lo/bound_hi
    on E[Y1−Y0 | always selected]."""
    yy = np.asarray(y, dtype=np.float64)
    d = np.asarray(treat, dtype=np.float64)
    s = np.asarray(selected, dtype=np.float64)
    n = yy.shape[0]
    if yy.ndim != 1 or d.shape != (n,) or s.shape != (n,) or n < 60:
        raise ValueError("matched (N,) arrays, N>=60 required")
    if not np.all(np.isfinite(yy)):
        raise ValueError("finite y required (pass selected rows only is fine)")
    if not np.all(np.isin(d, (0.0, 1.0))) or not np.all(np.isin(s, (0.0, 1.0))):
        raise ValueError("treat/selected must be binary")
    p1 = float(s[d == 1].mean())
    p0 = float(s[d == 0].mean())
    if min(p0, p1) <= 0 or max(p0, p1) >= 1:
        raise ValueError("both selection rates must be interior")
    if abs(p1 - p0) < 1e-9:
        # no differential selection: naive diff bounds itself
        diff = float(yy[(d == 1) & (s == 1)].mean() - yy[(d == 0) & (s == 1)].mean())
        return {"bound_lo": diff, "bound_hi": diff, "p1": p1, "p0": p0}
    if p1 > p0:
        over, under = 1.0, 0.0
        q = p0 / p1  # keep-share of the treated selected dist
    else:
        over, under = 0.0, 1.0
        q = p1 / p0
    y_over = yy[(d == over) & (s == 1)]
    y_under = yy[(d == under) & (s == 1)]
    if y_over.size < 10 or y_under.size < 10:
        raise ValueError("not enough selected observations")
    lo_q = float(np.quantile(y_over, 1 - q))
    hi_q = float(np.quantile(y_over, q))
    m_lo = float(y_over[y_over >= lo_q].mean())  # trim bottom
    m_hi = float(y_over[y_over <= hi_q].mean())  # trim top
    m_under = float(y_under.mean())
    if over == 1.0:
        bound_lo = m_lo - m_under
        bound_hi = m_hi - m_under
    else:
        bound_lo = m_under - m_hi
        bound_hi = m_under - m_lo
    return {
        "bound_lo": float(min(bound_lo, bound_hi)),
        "bound_hi": float(max(bound_lo, bound_hi)),
        "p1": p1,
        "p0": p0,
    }


def synth_lee_bounds(
    n: int = 4000,
    effect: float = 1.0,
    sel_shift: float = 0.4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Selection: S=1 iff latent index crosses arm-specific
    threshold → treated arm selects more (sel_shift). Effect
    on always-selected = effect."""
    rng = np.random.default_rng(seed)
    d = rng.binomial(1, 0.5, n).astype(np.float64)
    v = rng.normal(0, 1, n)  # latent selection propensity
    u = rng.normal(0, 1, n)
    s = (v + sel_shift * d > 0).astype(np.float64)
    y = effect * d + 0.6 * v + u
    return {"y": y, "treat": d, "selected": s}


def bench_lee_bounds(seed: int = 20261231 + 261) -> dict[str, float]:
    """Lee self-check: bounds bracket the true effect=1 while
    the naive diff is biased by selection (v enters y).
    All ``synthetic_*``."""
    d = synth_lee_bounds(effect=1.0, sel_shift=0.4, seed=seed)
    out = lee_bounds(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["selected"]),
    )
    yy = np.asarray(d["y"])
    dd = np.asarray(d["treat"])
    ss = np.asarray(d["selected"])
    naive = float(yy[(dd == 1) & (ss == 1)].mean() - yy[(dd == 0) & (ss == 1)].mean())
    out2 = lee_bounds(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["selected"]),
    )
    brackets = out["bound_lo"] <= 1.0 <= out["bound_hi"]
    return {
        "synthetic_bound_lo": out["bound_lo"],
        "synthetic_bound_hi": out["bound_hi"],
        "synthetic_naive": naive,
        "synthetic_p1": out["p1"],
        "synthetic_p0": out["p0"],
        "synthetic_true": 1.0,
        "synthetic_detects": float(brackets and abs(naive - 1.0) > 0.15),
        "synthetic_determinism": float(out2["bound_lo"] == out["bound_lo"]),
    }
