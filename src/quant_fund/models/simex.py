"""SIMEX — simulation-extrapolation measurement-error correction.

When a regressor is observed with additive classical measurement
error (W = X + U, known σ_u), the naive slope attenuates toward
zero. SIMEX re-adds increasing noise doses, re-estimates, and
extrapolates the attenuation curve back to zero error — recovering
the error-free coefficient without instruments.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure attenuation correction on
generated mismeasured panels — never market evidence.

References:
- Cook, J. R., Stefanski, L. A. (1994). Simulation-extrapolation
  estimation in parametric measurement error models. *JASA* 89,
  1314-1328 — the SIMEX procedure.
- Stefanski, L. A., Cook, J. R. (1995). Simulation-extrapolation:
  the measurement error jackknife. *JASA* 90, 1247-1256 — variance
  estimation via the pseudo-jackknife.
- Carroll, R. J., Ruppert, D., Stefanski, L. A., Crainiceanu, C. M.
  (2006). *Measurement Error in Nonlinear Models*, 2nd ed. Chapman
  & Hall — quadratic extrapolant recommendation and σ_u-knowledge
  assumptions.
- Fuller, W. A. (1987). *Measurement Error Models*. Wiley —
  reliability-ratio machinery underlying the benchmark comparison.

Composition: pure numpy — quadratic extrapolant over a λ-grid,
jackknife SE on the extrapolated point; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _ols_slope(y: FloatArray, w: FloatArray) -> float:
    wc = w - w.mean()
    return float(np.cov(wc, y)[0, 1] / np.var(wc))


def simex(
    y: FloatArray,
    w_obs: FloatArray,
    sigma_u: float,
    lambdas: FloatArray | None = None,
    n_boot: int = 50,
    seed: int = 0,
) -> dict[str, float]:
    """SIMEX-corrected slope estimate.

    ``w_obs = x + u`` with known measurement-error sd ``sigma_u``.
    For each λ in ``lambdas`` we simulate ``w + sqrt(λ)·σ_u·z``,
    estimate the slope, then extrapolate the quadratic
    slope-vs-(1+λ)σ_u² trend to zero added error."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    ww = np.asarray(w_obs, dtype=np.float64).ravel()
    n = yy.size
    if ww.size != n or n < 40:
        raise ValueError("equal-length y and w, n>=40")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(ww)):
        raise ValueError("finite y and w required")
    if sigma_u <= 0 or not math.isfinite(sigma_u):
        raise ValueError("sigma_u must be positive and finite")
    lams = (
        np.asarray(lambdas, dtype=np.float64).ravel()
        if lambdas is not None
        else np.array([0.5, 1.0, 1.5, 2.0])
    )
    if lams.size < 3 or np.any(lams <= 0):
        raise ValueError("need >=3 positive lambda doses")

    rng = np.random.default_rng(seed)
    naive = _ols_slope(yy, ww)
    extra_var = (1.0 + lams) * sigma_u**2  # total error variance per λ
    sim_slopes = np.zeros(lams.size)
    for i, lam in enumerate(lams):
        est = []
        for _ in range(n_boot):
            w2 = ww + math.sqrt(lam) * sigma_u * rng.standard_normal(n)
            est.append(_ols_slope(yy, w2))
        sim_slopes[i] = float(np.mean(est))

    # quadratic extrapolation to total error variance 0
    design = np.column_stack([np.ones(lams.size), extra_var, extra_var**2])
    coef = np.linalg.lstsq(design, sim_slopes, rcond=None)[0]
    corrected = float(coef[0])
    resid = sim_slopes - design @ coef
    r2 = float(1 - resid @ resid / max(float(((sim_slopes - sim_slopes.mean()) ** 2).sum()), 1e-30))

    # pseudo-jackknife SE: leave-one-out influence on the extrapolation
    jk = np.zeros(n)
    step = max(n // 25, 8)
    idx = np.arange(0, n, max(n // 20, 1))[:20]  # 20 pseudo-deletions
    for j, ix in enumerate(idx[:20]):
        keep = np.ones(n, dtype=bool)
        keep[ix : ix + step] = False
        sl = np.zeros(lams.size)
        for i, lam in enumerate(lams):
            est = []
            for _ in range(max(n_boot // 2, 10)):
                w2 = ww[keep] + math.sqrt(lam) * sigma_u * rng.standard_normal(int(keep.sum()))
                sl[i] += _ols_slope(yy[keep], w2)
            sl[i] /= max(n_boot // 2, 10)
        jk[j] = np.linalg.lstsq(design, sl, rcond=None)[0][0]
    se = float(
        np.std(jk[: len(idx[:20])], ddof=1)
        * math.sqrt(len(idx[:20]) - 1)
        / math.sqrt(len(idx[:20]))
    )

    # attenuation ratio: naive/corrected (reliability proxy)
    atten = float(abs(naive / corrected)) if abs(corrected) > 1e-9 else math.nan
    z = corrected / max(se, 1e-9)
    return {
        "naive": naive,
        "corrected": corrected,
        "se": se,
        "z": float(z),
        "p_value": float(2 * (1 - norm.cdf(abs(z)))),
        "attenuation": atten,
        "extrapolation_r2": r2,
        "sigma_u": float(sigma_u),
        "n": float(n),
    }


def synth_mismeasured(
    n: int = 1500,
    beta: float = 1.0,
    reliability: float = 0.6,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Mismeasurement DGP: w = x + u with ``reliability`` =
    Var(x)/Var(w); lower reliability → stronger attenuation."""
    if not 0.2 < reliability < 0.98:
        raise ValueError("reliability in (0.2, 0.98)")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    su = math.sqrt((1.0 - reliability) / reliability)
    w = x + su * rng.standard_normal(n)
    y = beta * x + rng.normal(0.0, 0.4, n)
    return {
        "y": y,
        "w": w,
        "sigma_u": np.array([su]),
        "beta_true": np.array([beta]),
    }


def bench_simex(seed: int = 20261231 + 213) -> dict[str, float]:
    """SIMEX self-check: corrected slope beats naive under heavy
    mismeasurement. All ``synthetic_*``."""
    d = synth_mismeasured(beta=1.0, reliability=0.6, seed=seed)
    out = simex(
        np.asarray(d["y"]),
        np.asarray(d["w"]),
        float(np.asarray(d["sigma_u"])[0]),
        seed=seed,
    )
    d0 = synth_mismeasured(beta=0.0, reliability=0.6, seed=seed + 1)
    out0 = simex(
        np.asarray(d0["y"]),
        np.asarray(d0["w"]),
        float(np.asarray(d0["sigma_u"])[0]),
        seed=seed + 1,
    )
    out_b = simex(
        np.asarray(d["y"]),
        np.asarray(d["w"]),
        float(np.asarray(d["sigma_u"])[0]),
        seed=seed,
    )

    corr = float(out["corrected"])
    naive = float(out["naive"])
    return {
        "synthetic_corrected": corr,
        "synthetic_corrected_err": float(abs(corr - 1.0)),
        "synthetic_naive": naive,
        "synthetic_naive_err": float(abs(naive - 1.0)),
        "synthetic_beats_naive": float(abs(corr - 1.0) < abs(naive - 1.0)),
        "synthetic_attenuation": float(out["attenuation"]),
        "synthetic_extrap_r2": float(out["extrapolation_r2"]),
        "synthetic_null_corrected": float(abs(out0["corrected"])),
        "synthetic_detects": float(abs(corr - 1.0) < 0.25 and abs(float(out0["corrected"])) < 0.3),
        "synthetic_determinism": float(corr == float(out_b["corrected"])),
    }
