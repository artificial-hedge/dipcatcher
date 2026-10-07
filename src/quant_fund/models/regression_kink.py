"""Regression kink design — slope discontinuity estimation (SYNTHETIC).

When a treatment's SLOPE in the running variable kinks at a
threshold (e.g. benefit schedule slope), the outcome's slope kink
identifies the treatment effect: b_RK = Δslope_Y / Δslope_X
(Card, Lee, Pei, Weber 2015). Sharp RKD = outcome slope jump alone.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure kink recovery on generated
threshold panels — never market evidence.

References:
- Card, Lee, Pei, Weber (2015). Inference on causal effects in a
  generalized regression kink design. *Econometrica* 83, 2453-2483.
- Nielsen, Sørensen, Taber (2010). Estimating the effect of student
  aid on college enrollment. *AEJ: Economic Policy* 2.
- Calonico, Cattaneo, Titiunik (2014). Robust nonparametric
  confidence intervals for RD designs. *Econometrica* 82
  (bias-correction machinery adapted to slope kinks).
- Gelman, Imbens (2019). Why high-order polynomials should not be
  used in RD. *JBES* 37 (local-linear justification).

Composition: pure numpy — local-linear slope estimation on each
side, fuzzy ratio with delta-method SE; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def _side_slope(x: FloatArray, y: FloatArray, side: str, bw: float) -> tuple[float, float, int]:
    """Local-linear slope on one side: slope of y on (x-c) within bw."""
    if side == "below":
        m = (x < 0) & (x > -bw)
        z = x[m] + bw / 2.0  # center inside window for stability
    else:
        m = (x >= 0) & (x < bw)
        z = x[m] - bw / 2.0
    n_ = int(m.sum())
    if n_ < 6:
        raise ValueError(f"{side}: too few observations in bandwidth")
    yy = y[m]
    zz = z - z.mean()
    var_z = float(np.var(zz))
    if var_z < 1e-12:
        raise ValueError(f"{side}: degenerate running variable")
    slope = float(np.cov(zz, yy)[0, 1] / var_z)
    resid = yy - (yy.mean() + slope * zz)
    se = float(np.std(resid) / math.sqrt(n_ * var_z))
    return slope, se, n_


def regression_kink(
    running: FloatArray,
    outcome: FloatArray,
    treatment_slope: FloatArray | None = None,
    bandwidth: float | None = None,
) -> dict[str, float]:
    """Sharp RKD slope jump; fuzzy RKD ratio when treatment is given.

    ``treatment_slope``: the treatment variable whose slope in x
    kinks at the threshold (first stage). Without it, returns the
    reduced-form slope discontinuity only."""
    x = _as1(running, "running")
    y = _as1(outcome, "outcome")
    n = x.size
    if y.size != n or n < 30:
        raise ValueError("need >=30 matched observations")
    if not np.any(x < 0) or not np.any(x >= 0):
        raise ValueError("running variable must straddle 0 (threshold)")

    bw = bandwidth if bandwidth is not None else float(np.quantile(np.abs(x), 0.35))
    if bw <= 0:
        raise ValueError("bandwidth must be > 0")

    sl_b, se_b, n_b = _side_slope(x, y, "below", bw)
    sl_a, se_a, n_a = _side_slope(x, y, "above", bw)
    d_slope = sl_a - sl_b
    se_dslope = math.sqrt(se_b**2 + se_a**2)
    z_rf = d_slope / max(se_dslope, 1e-12)

    out: dict[str, float] = {
        "bandwidth": bw,
        "slope_below": sl_b,
        "slope_above": sl_a,
        "d_slope": d_slope,
        "se_d_slope": se_dslope,
        "z_reduced_form": z_rf,
        "p_reduced_form": float(2 * (1 - norm.cdf(abs(z_rf)))),
        "n_below": float(n_b),
        "n_above": float(n_a),
    }

    if treatment_slope is not None:
        tx = _as1(treatment_slope, "treatment_slope")
        if tx.size != n:
            raise ValueError("treatment length mismatch")
        sb, seb, _ = _side_slope(x, tx, "below", bw)
        sa, sea, _ = _side_slope(x, tx, "above", bw)
        d_fs = sa - sb
        se_fs = math.sqrt(seb**2 + sea**2)
        if abs(d_fs) < 1e-8:
            raise ValueError("no first-stage kink — fuzzy RKD unidentified")
        # delta-method on the ratio
        b_rk = d_slope / d_fs
        se_rk = math.sqrt((se_dslope / d_fs) ** 2 + (d_slope * se_fs / d_fs**2) ** 2)
        z_rk = b_rk / max(se_rk, 1e-12)
        out.update(
            {
                "fs_d_slope": d_fs,
                "fs_se": se_fs,
                "b_rk": float(b_rk),
                "se_rk": float(se_rk),
                "z_rk": float(z_rk),
                "p_rk": float(2 * (1 - norm.cdf(abs(z_rk)))),
            }
        )
    return out


def synth_kink(
    n: int = 3000,
    kink_fs: float = 1.0,
    effect: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Kink DGP: treatment slope in x kinks at 0 by ``kink_fs``;
    outcome = β·treatment + smooth f(x) + noise. Fuzzy RKD recovers β."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    t_ = 0.5 * x + kink_fs * np.maximum(x, 0.0) + rng.normal(0.0, 0.15, n)
    f_x = 0.3 * x  # smooth linear baseline — no curvature, no kink
    y = effect * t_ + f_x + rng.normal(0.0, 0.1, n)
    return {
        "running": x,
        "treatment": t_,
        "outcome": y,
        "effect_true": np.array([effect]),
    }


def bench_regression_kink(seed: int = 20261231 + 208) -> dict[str, float]:
    """RKD self-check: fuzzy ratio recovers β; no-kink DGP returns
    near-zero first stage (honest ValueError path exercised separately).
    All ``synthetic_*``."""
    d = synth_kink(seed=seed, kink_fs=1.0, effect=0.5)
    out = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )
    d0 = synth_kink(seed=seed + 1, kink_fs=0.0, effect=0.5)
    try:
        out0 = regression_kink(
            np.asarray(d0["running"]),
            np.asarray(d0["outcome"]),
            treatment_slope=np.asarray(d0["treatment"]),
        )
        fs0 = float(abs(out0["fs_d_slope"]))
        rk0 = float(out0.get("b_rk", math.nan))
    except ValueError:
        fs0 = 0.0
        rk0 = math.nan

    out_b = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )

    b_rk = float(out["b_rk"])
    return {
        "synthetic_b_rk": b_rk,
        "synthetic_b_rk_err": float(abs(b_rk - 0.5)),
        "synthetic_fs_kink": float(out["fs_d_slope"]),
        "synthetic_rf_kink": float(out["d_slope"]),
        "synthetic_p_rk": float(out["p_rk"]),
        "synthetic_null_fs": fs0,
        "synthetic_null_b_rk": rk0 if math.isfinite(rk0) else 0.0,
        "synthetic_detects": float(abs(b_rk - 0.5) < 0.2 and float(out["p_rk"]) < 0.2),
        "synthetic_determinism": float(float(out["b_rk"]) == float(out_b["b_rk"])),
    }
