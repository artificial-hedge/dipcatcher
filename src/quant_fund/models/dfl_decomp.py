"""DiNardo-Fortin-Lemieux (1996) reweighting decomposition.

Where Oaxaca-Blinder decomposes mean gaps, DFL decomposes
the whole distribution: reweight group B by the fitted odds
of belonging to A, yielding the counterfactual distribution
"A-structured outcomes under B covariates" (or the reverse).
The mean gap splits into a structure term and a composition
term consistent at every quantile.

Honesty: synthetic groups embed known composition and
structure shifts; the bench checks the decomposition signs
and magnitudes — proper diagnostics, never market evidence.

References:
- DiNardo, J., Fortin, N. M., Lemieux, T. (1996). Labor
  market institutions and the distribution of wages,
  1973-1992: a semiparametric approach. *Econometrica* 64 —
  the reweighting.
- Firpo, S., Fortin, N. M., Lemieux, T. (2009).
  Unconditional quantile regressions. *Econometrica* 77 —
  the RIF complement (wave 34).
- Barsky, R., Bound, J., Charles, K. K., Lupton, J. P.
  (2002). Accounting for the Black-White wealth gap.
  *Journal of the American Statistical Association* 97 —
  application.
- Fortin, N., Lemieux, T., Firpo, S. (2011). Decomposition
  methods in economics. *Handbook of Labor Economics* 4A —
  survey.

Composition: pure numpy — logit propensity + weighted
quantiles; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _wmean(y: FloatArray, w: FloatArray) -> float:
    return float(np.sum(y * w) / np.sum(w))


def _wq(y: FloatArray, w: FloatArray, q: FloatArray) -> FloatArray:
    """Weighted quantiles at probabilities q."""
    o = np.argsort(y)
    ys, ws = y[o], w[o]
    cw = np.cumsum(ws) / ws.sum()
    return np.asarray(np.interp(q, cw, ys), dtype=np.float64)


def dfl_decompose(
    y: FloatArray,
    x: FloatArray,
    group: FloatArray,
    quantiles: FloatArray | None = None,
) -> dict[str, float]:
    """DFL decomposition. group ∈ {0,1}; x includes a
    constant. Counterfactual C = group-B outcomes reweighted
    to A's covariate distribution."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    g = np.asarray(group, dtype=np.float64)
    n = yy.shape[0]
    if xx.ndim != 2 or xx.shape[0] != n or g.shape != (n,) or n < 60:
        raise ValueError("y (N,), x (N,k), group (N,), N>=60")
    if not np.all(np.isin(g, (0.0, 1.0))):
        raise ValueError("group must be binary")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    a, b = g == 1.0, g == 0.0
    if a.sum() < 20 or b.sum() < 20:
        raise ValueError("each group needs >=20 obs")
    # propensity: logit P(A|x) via IRLS-lite (simple Newton)
    z = xx
    coef = np.zeros(z.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-z @ coef))
        w = p * (1 - p) + 1e-9
        grad = z.T @ (g - p)
        hess = z.T @ (w[:, None] * z)
        step = np.linalg.solve(hess, grad)
        coef += step
        if np.max(np.abs(step)) < 1e-8:
            break
    pa = 1.0 / (1.0 + np.exp(-z @ coef))
    # reweight B to look like A: w_B = p/(1-p)
    wb = pa[b] / np.clip(1 - pa[b], 1e-6, 1.0)
    wb = np.clip(wb, 0, np.quantile(wb, 0.99))
    ya, yb = yy[a], yy[b]
    wa = np.ones(a.sum())
    gap = float(ya.mean() - yb.mean())
    cf = _wmean(yb, wb)  # counterfactual B outcomes at A's x
    # composition = cf − B (gap from the x distribution);
    # structure = A − cf (gap from different returns)
    composition = float(cf - yb.mean())
    structure = float(ya.mean() - cf)
    qs = quantiles if quantiles is not None else np.array([0.25, 0.5, 0.75])
    qq = np.asarray(qs, dtype=np.float64)
    qa = _wq(ya, wa, qq)
    qb = _wq(yb, np.ones(b.sum()), qq)
    qc = _wq(yb, wb, qq)
    out: dict[str, float] = {
        "gap": gap,
        "structure": structure,
        "composition": composition,
    }
    for i, qv in enumerate(qq):
        out[f"q{qv:.2f}_a"] = float(qa[i])
        out[f"q{qv:.2f}_b"] = float(qb[i])
        out[f"q{qv:.2f}_cf"] = float(qc[i])
    return out


def synth_dfl(
    n: int = 3000,
    comp_shift: float = 0.8,
    struct_shift: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Group A: x shifted (composition) and intercept shifted
    (structure)."""
    rng = np.random.default_rng(seed)
    g = rng.binomial(1, 0.5, n).astype(np.float64)
    x1 = rng.normal(0, 1, n) + comp_shift * g
    y = 1.5 * x1 + struct_shift * g + rng.normal(0, 1, n)
    return {
        "y": y,
        "x": np.column_stack([np.ones(n), x1]),
        "group": g,
    }


def bench_dfl_decomp(seed: int = 20261231 + 268) -> dict[str, float]:
    """DFL self-check: structure ≈ struct_shift at the mean,
    composition captures the x-shift; counterfactual median
    moves toward A. All ``synthetic_*``."""
    d = synth_dfl(comp_shift=0.8, struct_shift=0.5, seed=seed)
    out = dfl_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    d2 = synth_dfl(comp_shift=0.8, struct_shift=0.5, seed=seed)
    out2 = dfl_decompose(np.asarray(d2["y"]), np.asarray(d2["x"]), np.asarray(d2["group"]))
    return {
        "synthetic_gap": out["gap"],
        "synthetic_structure": out["structure"],
        "synthetic_composition": out["composition"],
        "synthetic_true_structure": 0.5,
        "synthetic_q50_cf": out["q0.50_cf"],
        "synthetic_q50_b": out["q0.50_b"],
        "synthetic_detects": float(
            abs(out["structure"] - 0.5) < 0.25
            and out["composition"] > 0.8
            and abs(out["structure"] + out["composition"] - out["gap"]) < 1e-9
        ),
        "synthetic_determinism": float(out2["gap"] == out["gap"]),
    }
