"""Roy model of self-selection across two sectors (SYNTHETIC).

Agents choose the sector paying them more; observed wages are
then truncated mixtures of sector-specific skill distributions.
The two-step estimator runs a selection probit on observables,
builds per-sector inverse-Mills terms, and re-estimates the
sector wage equations with the Mills correction — recovering
sector skill prices and the selection-correlation pattern
(positive vs negative selection).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure skill-price recovery on
generated sector-choice panels — never market evidence.

References:
- Roy, A. D. (1951). Some thoughts on the distribution of
  earnings. *Oxford Economic Papers* 3, 135-146 — comparative
  advantage self-selection.
- Heckman, J. J., Honore, B. E. (1990). The empirical content
  of the Roy model. *Econometrica* 58 — identification of
  sector skill prices under selection.
- Borjas, G. J. (1987). Self-selection and the earnings of
  immigrants. *American Economic Review* 77 — positive vs
  negative selection diagnostics.
- French, E., Taber, C. (2011). Identification of models of
  the labor market. *Handbook of Labor Economics* 4.

Composition: pure numpy + scipy — probit IRLS selection,
per-sector Mills-augmented OLS; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _probit(x: FloatArray, s: FloatArray, iters: int = 40) -> FloatArray:
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        xb = x @ b
        p = np.clip(stats.norm.cdf(xb), 1e-9, 1 - 1e-9)
        w = stats.norm.pdf(xb) ** 2 / (p * (1 - p))
        grad = x.T @ ((s - p) * stats.norm.pdf(xb) / (p * (1 - p)))
        h = (x * w[:, None]).T @ x
        try:
            step = np.linalg.solve(h + 1e-8 * np.eye(h.shape[0]), grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular probit Hessian") from exc
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return np.asarray(b, dtype=np.float64)


def _mills(z: FloatArray) -> FloatArray:
    return np.asarray(
        stats.norm.pdf(z) / np.clip(stats.norm.cdf(z), 1e-9, None),
        dtype=np.float64,
    )


def roy_fit(
    wage: FloatArray,
    sector: FloatArray,
    x: FloatArray,
) -> dict[str, float]:
    """Two-step Roy fit.

    wage (log), sector ∈ {0,1} (chosen), x observables (n×k,
    no intercept): x[:,0] enters the wage equations; columns
    1..k−1 additionally enter selection only (exclusion).
    With k=1 both use x[:,0] (fragile without exclusion)."""
    w = np.asarray(wage, dtype=np.float64).ravel()
    s = np.asarray(sector, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    n = w.size
    if xx.ndim != 2 or xx.shape[0] != n or s.size != n:
        raise ValueError("aligned wage/sector/x required")
    if n < 200:
        raise ValueError("n>=200")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    if set(np.unique(s)) - {0.0, 1.0}:
        raise ValueError("sector in {0,1}")
    n1 = int(np.sum(s == 1.0))
    n0 = n - n1
    if min(n0, n1) < 80:
        raise ValueError("both sectors need >=80 obs")

    xs = np.column_stack([np.ones(n), xx])
    b_sel = _probit(xs, s)
    zhat = xs @ b_sel
    lam1 = _mills(zhat)  # sector 1 correction
    lam0 = _mills(-zhat)  # sector 0 correction (of -z)

    i1 = s == 1.0
    i0 = ~i1
    xw = xx[:, 0]
    x1 = np.column_stack([np.ones(n1), xw[i1], lam1[i1]])
    x0 = np.column_stack([np.ones(n0), xw[i0], -lam0[i0]])
    b1, *_ = np.linalg.lstsq(x1, w[i1], rcond=None)
    b0, *_ = np.linalg.lstsq(x0, w[i0], rcond=None)

    # naive sector wage gap vs selection-corrected gap
    gap_naive = float(np.mean(w[i1]) - np.mean(w[i0]))
    gap_corrected = float(b1[0] - b0[0])
    rho1 = float(b1[-1])  # coefficient on mills ≈ σ_e·ρ
    rho0 = float(-b0[-1])

    return {
        "n": float(n),
        "share_sector1": float(n1 / n),
        "skill_price1": float(b1[1]),
        "skill_price0": float(b0[1]),
        "gap_naive": gap_naive,
        "gap_corrected": gap_corrected,
        "rho1_mills": rho1,
        "rho0_mills": rho0,
        "sel_coef0": float(b_sel[1]),
        "intercept1": float(b1[0]),
        "intercept0": float(b0[0]),
    }


def synth_roy(
    n: int = 1200,
    price1: float = 0.5,
    price0: float = 0.3,
    corr: float = 0.4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Two-sector selectivity: sector index c·x+η>0; wage
    equations share x with sector errors correlated with η."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    # selection index driven by x plus unobservable η correlated
    # with both sector errors (selectivity, not pure Roy sort)
    c = 0.6
    z = rng.normal(0.0, 1.0, n)  # exclusion: selection only
    e = rng.multivariate_normal(
        [0, 0, 0],
        [[1, corr, 0.5], [corr, 1, 0.3], [0.5, 0.3, 1]],
        n,
    )
    sector = (c * x + 0.5 * z + e[:, 2] > 0).astype(np.float64)
    logw1 = 0.2 + price1 * x + e[:, 0]
    logw0 = -0.1 + price0 * x + e[:, 1]
    wage = np.where(sector == 1.0, logw1, logw0)
    xx = np.column_stack([x, z])
    return {"wage": wage, "sector": sector, "x": xx}


def bench_roy_model(seed: int = 20261231 + 243) -> dict[str, float]:
    """Roy self-check: sector-1 skill price recovered near .5,
    sector-0 near .3, positive selection (ρ1_mills > 0), and the
    naive wage gap overstates the corrected gap. All
    ``synthetic_*``."""
    d = synth_roy(price1=0.5, price0=0.3, seed=seed)
    out = roy_fit(d["wage"], d["sector"], d["x"])
    out_b = roy_fit(d["wage"], d["sector"], d["x"])

    p1 = float(out["skill_price1"])
    return {
        "synthetic_skill_price1": p1,
        "synthetic_skill_price0": float(out["skill_price0"]),
        "synthetic_rho1": float(out["rho1_mills"]),
        "synthetic_rho0": float(out["rho0_mills"]),
        "synthetic_gap_naive": float(out["gap_naive"]),
        "synthetic_gap_corrected": float(out["gap_corrected"]),
        "synthetic_share1": float(out["share_sector1"]),
        "synthetic_detects": float(
            0.35 < p1 < 0.7
            and 0.0 < float(out["skill_price0"]) < 0.55
            and float(out["rho1_mills"]) > 0.2
            and float(out["gap_naive"]) > float(out["gap_corrected"]) + 0.3
            and 0.3 < float(out["share_sector1"]) < 0.7
        ),
        "synthetic_determinism": float(p1 == float(out_b["skill_price1"])),
    }
