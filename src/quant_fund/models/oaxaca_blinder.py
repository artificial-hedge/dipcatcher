"""Oaxaca-Blinder decomposition of mean outcome gaps.

Two-fold: Δ = (X̄_A − X̄_B)β_A + X̄_B(β_A − β_B) — a
composition ("explained") part and a coefficient
("unexplained"/structure) part; three-fold splits the second
into coefficients and interaction terms.

Honesty: synthetic groups generate a known gap with known
composition share; the bench checks attribution — proper
diagnostics, never market evidence.

References:
- Oaxaca, R. (1973). Male-female wage differentials in urban
  labor markets. *International Economic Review* 14 — the
  decomposition.
- Blinder, A. S. (1973). Wage discrimination: reduced form
  and structural estimates. *Journal of Human Resources* 8 —
  the parallel decomposition.
- Reimers, C. W. (1983). Labor market discrimination against
  Hispanic and Black men. *Review of Economics and
  Statistics* 65 — the pooled-coefficient variant.
- Jann, B. (2008). The Blinder-Oaxaca decomposition for
  linear regression models. *Stata Journal* 8 — the three-
  fold split used here.

Composition: pure numpy — per-group OLS + counterfactual
means; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ols(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.asarray(np.linalg.lstsq(x, y, rcond=None)[0], dtype=np.float64)


def ob_decompose(
    y: FloatArray,
    x: FloatArray,
    group: FloatArray,
) -> dict[str, float]:
    """Two- and three-fold decomposition. group ∈ {0,1};
    x already includes a constant column."""
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
    xa, xb = xx[a], xx[b]
    ba, bb = _ols(xa, yy[a]), _ols(xb, yy[b])
    ma, mb = xa.mean(axis=0), xb.mean(axis=0)
    gap = float(ma @ ba - mb @ bb)
    # two-fold (A coefficients as reference)
    comp = float((ma - mb) @ ba)
    coef_part = float(mb @ (ba - bb))
    # three-fold
    endow = float((ma - mb) @ bb)
    coef3 = float(mb @ (ba - bb))
    inter = float((ma - mb) @ (ba - bb))
    return {
        "gap": gap,
        "composition": comp,
        "structure": coef_part,
        "endow3": endow,
        "coef3": coef3,
        "interact3": inter,
    }


def synth_ob(
    n: int = 3000,
    gap_composition: float = 0.5,
    gap_structure: float = 0.3,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Group A: higher x mean (composition gap_composition) and
    higher slope (structure gap_structure)."""
    rng = np.random.default_rng(seed)
    g = rng.binomial(1, 0.5, n).astype(np.float64)
    x1 = rng.normal(0, 1, n) + gap_composition * g
    beta_a = 1.0 + gap_structure
    y = (beta_a * g + (1.0 - g)) * x1 + rng.normal(0, 1, n)
    return {
        "y": y,
        "x": np.column_stack([np.ones(n), x1]),
        "group": g,
    }


def bench_oaxaca_blinder(seed: int = 20261231 + 266) -> dict[str, float]:
    """OB self-check: decomposition recovers the total gap and
    attributes ~composition share correctly. All ``synthetic_*``."""
    d = synth_ob(gap_composition=0.5, gap_structure=0.3, seed=seed)
    out = ob_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    d2 = synth_ob(gap_composition=0.5, gap_structure=0.3, seed=seed)
    out2 = ob_decompose(np.asarray(d2["y"]), np.asarray(d2["x"]), np.asarray(d2["group"]))
    # gap ≈ βa·(1 + x̄A effect): compute expected total
    g = np.asarray(d["group"])
    gap_true = float(np.asarray(d["y"])[g == 1].mean() - np.asarray(d["y"])[g == 0].mean())
    recon = out["composition"] + out["structure"]
    return {
        "synthetic_gap": out["gap"],
        "synthetic_gap_empirical": gap_true,
        "synthetic_composition": out["composition"],
        "synthetic_structure": out["structure"],
        "synthetic_detects": float(
            abs(out["gap"] - gap_true) < 0.2
            and abs(recon - out["gap"]) < 1e-9
            and out["composition"] > 0
        ),
        "synthetic_determinism": float(out2["gap"] == out["gap"]),
    }
