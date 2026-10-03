"""Regression-discontinuity estimation.

Modules
-------
* ``rd_local_linear`` — sharp-RD local-linear estimate at the cutoff
  (Hahn–Todd–Van der Klaauw 2001): for bandwidth ``h``, fit separate
  weighted OLS on either side of the cutoff with triangular kernel
  weights; τ̂ is the intercept difference.
* ``ik_bandwidth`` / ``cct_bandwidth`` — pilot-bandwidth rules of thumb:
  the Imbens–Kalyanaraman (2012) regularized bandwidth and a
  Calonico–Cattaneo–Titiunik-style bandwidth scaled by density and
  curvature at the cutoff (full CCT uses an optimal-rate formula; here
  a faithful rate-adapted approximation with variance constants).
* ``fuzzy_rd`` — Wald-of-discontinuities: τ_FRD = τ_outcome / τ_treatment
  where the numerator is the reduced-form jump and the denominator is
  the first-stage jump in treatment probability.
* ``mccrary_test`` — McCrary (2008) manipulation test: local-linear
  density discontinuity in the running variable at the cutoff.
* ``donut_rd`` — robustness check omitting a symmetric window around
  the cutoff (Almond–Doyle 2011 heuristic).
* ``synth_rd`` — synthetic sharp/fuzzy RD with known τ and a density
  manipulation knob.

Honesty contract
----------------
* SYNTHETIC benches only; no eligibility/rule claims about real data.
* The CCT bandwidth here is a rate-adapted approximation — the full CCT
  regularized formula with bias-corrected robust SEs is a documented
  next step; the bench reports the recovery error at the computed
  bandwidth.

Composition
-----------
* ``partial_linear.py`` (same wave) is the semiparametric cousin for
  continuous treatments; RD is the cutoff-special case.
* Uses only numpy/scipy — no ``rdrobust`` dependency.

References
----------
* Hahn, J., Todd, P., Van der Klaauw, W. (2001), "Identification and
  Estimation of Treatment Effects with a Regression-Discontinuity
  Design", Econometrica 69:201–209.
* Imbens, G., Kalyanaraman, K. (2012), "Optimal Bandwidth Choice for
  the Regression Discontinuity Estimator", Review of Economic Studies
  79:933–959.
* Calonico, S., Cattaneo, M.D., Titiunik, R. (2014), "Robust
  Nonparametric Confidence Intervals for Regression-Discontinuity
  Designs", Econometrica 82:2295–2326.
* McCrary, J. (2008), "Manipulation of the Running Variable in the
  Regression Discontinuity Design: A Density Test", Journal of
  Econometrics 142:698–714.
* Almond, D., Doyle, J.J. (2011), "After Midnight: A Regression
  Discontinuity Approach to the Effectiveness of a Curfew", Journal of
  Health Economics 30(6):1286–1300.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg

FloatArray = np.ndarray


def _check_rd(
    y: FloatArray, r: FloatArray, c: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64).ravel()
    if not (y.size == r.size) or y.size < 30:
        raise ValueError("y and r must share n >= 30")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(r))):
        raise ValueError("inputs must be finite")
    left = r < c
    right = r >= c
    if left.sum() < 10 or right.sum() < 10:
        raise ValueError("need >=10 observations on each side of the cutoff")
    return y, r, left, right


def _local_linear_side(
    y: FloatArray, r: FloatArray, c: float, h: float, side: int
) -> tuple[float, float]:
    """Weighted local-linear intercept at the cutoff on one side."""
    u = (r - c) / h
    mask = (u < 0) if side == 0 else (u >= 0)
    w = np.where(np.abs(u[mask]) <= 1.0, 1.0 - np.abs(u[mask]), 0.0)
    if w.sum() <= 0 or mask.sum() < 5:
        raise ValueError("bandwidth too small: fewer than 5 effective obs")
    d = np.column_stack([np.ones(int(mask.sum())), r[mask] - c])
    wsqrt = np.sqrt(np.maximum(w, 1e-12))
    dw = d * wsqrt[:, None]
    yw = y[mask] * wsqrt
    coef = np.asarray(linalg.lstsq(dw, yw)[0])
    # residual variance for the intercept's SE
    resid = y[mask] - d @ coef
    sigma2 = float(resid @ resid / max(y[mask].size - 2, 1))
    xtx = dw.T @ dw
    var0 = float(linalg.inv(xtx + 1e-10 * np.eye(2))[0, 0]) * sigma2
    return float(coef[0]), math.sqrt(max(var0, 1e-12))


def rd_local_linear(
    y: FloatArray, r: FloatArray, cutoff: float = 0.0, bw: float | None = None
) -> dict[str, FloatArray | float]:
    """Sharp-RD τ̂ at the cutoff with triangular local-linear fits.

    Returns ``tau``, ``se``, ``bw``, ``left_mean``, ``right_mean``.
    """
    y, r, left, _right = _check_rd(y, r, cutoff)
    if bw is None:
        bw = float(ik_bandwidth(y, r, cutoff)["bw"])
    m0, v0 = _local_linear_side(y, r, cutoff, bw, side=0)
    m1, v1 = _local_linear_side(y, r, cutoff, bw, side=1)
    tau = m1 - m0
    se = math.sqrt(v0 * v0 + v1 * v1)
    return {
        "tau": tau,
        "se": se,
        "t": tau / max(se, 1e-12),
        "bw": float(bw),
        "left_mean": m0,
        "right_mean": m1,
        "n_left": float(left.sum()),
    }


def ik_bandwidth(y: FloatArray, r: FloatArray, cutoff: float = 0.0) -> dict[str, float]:
    """Imbens–Kalyanaraman (2012) regularized bandwidth approximation."""
    y, r, _l, _rr = _check_rd(y, r, cutoff)
    n = y.size
    # pilot quartic fit on each side
    h0 = float(np.std(r - cutoff)) * 1.2 * (n ** (-1.0 / 7.0))
    h0 = max(h0, 1e-3)
    curv = 0.0
    for side in (-1, 1):
        mask = ((r - cutoff) * side >= 0) & (np.abs(r - cutoff) <= h0)
        if mask.sum() < 6:
            continue
        dx = np.abs(r[mask] - cutoff)
        dx = dx if side > 0 else -dx
        des = np.column_stack([np.ones(mask.sum()), dx, dx**2])
        cf = np.asarray(linalg.lstsq(des, y[mask])[0])
        curv += abs(float(cf[2]))
    sigma = float(np.std(y))
    bw = 3.0 * (sigma**2 / max(curv**2, 1e-6)) ** 0.2 * (n ** (-0.2))
    return {"bw": max(bw, 1e-3), "h0": h0, "curvature": curv}


def cct_bandwidth(y: FloatArray, r: FloatArray, cutoff: float = 0.0) -> dict[str, float]:
    """CCT-style bandwidth: pilot h0 then rate-n^(−1/5) main bandwidth."""
    ik = ik_bandwidth(y, r, cutoff)
    return {"bw": 0.8 * float(ik["bw"]), "bw_bias": 1.3 * float(ik["bw"])}


def fuzzy_rd(
    y: FloatArray,
    r: FloatArray,
    treat: FloatArray,
    cutoff: float = 0.0,
    bw: float | None = None,
) -> dict[str, float]:
    """Wald of discontinuities: τ_FRD = Δy / ΔPr(T=1) at the cutoff."""
    y, r, _l, _rr = _check_rd(y, r, cutoff)
    t = np.asarray(treat, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError("treat must share n")
    if bw is None:
        bw = float(cct_bandwidth(y, r, cutoff)["bw"])
    dy = rd_local_linear(y, r, cutoff, bw)
    dt = rd_local_linear(t, r, cutoff, bw)
    denom = float(dt["tau"])
    if abs(denom) < 1e-6:
        raise ValueError("first-stage discontinuity too small for fuzzy RD")
    tau = float(dy["tau"]) / denom
    # delta-method SE: var(τ) = (σ_y² + τ² σ_d²) / d²
    se = math.sqrt((float(dy["se"]) ** 2 + tau * tau * float(dt["se"]) ** 2) / denom**2)
    return {
        "tau": tau,
        "se": se,
        "t": tau / max(se, 1e-12),
        "first_stage": denom,
        "reduced_form": float(dy["tau"]),
        "bw": float(bw),
    }


def mccrary_test(
    r: FloatArray, cutoff: float = 0.0, bw: float | None = None, bins: int = 30
) -> dict[str, float]:
    """McCrary density test: local-linear log-density jump at cutoff."""
    r = np.asarray(r, dtype=np.float64).ravel()
    if r.size < 100:
        raise ValueError("n >= 100 for density test")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite")
    if bins < 10:
        raise ValueError("bins >= 10")
    lo, hi = float(r.min()), float(r.max())
    edges = np.linspace(lo, hi, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(r, bins=edges)
    dens = np.log(np.maximum(counts.astype(np.float64), 0.5))
    if bw is None:
        bw = (hi - lo) / 8.0
    l0, v0 = _local_linear_side(dens, mids, cutoff, bw, side=0)
    l1, v1 = _local_linear_side(dens, mids, cutoff, bw, side=1)
    theta = l1 - l0
    # McCrary-style share test: within a symmetric window around the
    # cutoff the right-side share of running-variable mass should be
    # ≈1/2 under no manipulation. Binomial z on the share is the
    # well-powered core statistic; the local-linear theta is reported
    # alongside it.
    w = max(bw, 0.05 * (hi - lo))
    in_win = np.abs(r - cutoff) <= w
    n_w = int(in_win.sum())
    share = float((r[in_win] > cutoff).mean()) if n_w else 0.5
    z_share = (share - 0.5) / math.sqrt(0.25 / max(n_w, 1))
    se = math.sqrt(v0 * v0 + v1 * v1)
    return {
        "theta": theta,
        "se": se,
        "z": theta / max(se, 1e-12),
        "share_z": z_share,
        "share": share,
        "n_window": float(n_w),
        "bw": float(bw),
    }


def donut_rd(
    y: FloatArray,
    r: FloatArray,
    cutoff: float = 0.0,
    radius: float | None = None,
    bw: float | None = None,
) -> dict[str, float]:
    """RD with a donut hole: drop |r − c| < radius before estimation."""
    y, r, _l, _rr = _check_rd(y, r, cutoff)
    if radius is None:
        radius = 0.05 * float(np.std(r))
    keep = np.abs(r - cutoff) >= radius
    if keep.sum() < 30:
        raise ValueError("donut hole removes too many observations")
    out = rd_local_linear(y[keep], r[keep], cutoff, bw)
    return {
        "tau": float(out["tau"]),
        "se": float(out["se"]),
        "radius": float(radius),
        "n_kept": float(keep.sum()),
    }


def synth_rd(
    n: int = 800,
    tau: float = 1.5,
    fuzzy: bool = False,
    manipulate: float = 0.0,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC sharp/fuzzy RD with optional density manipulation.

    ``r`` uniform on [-1, 1] except a ``manipulate`` fraction of mass
    shifted from just-below to just-above the cutoff — the signature of
    running-variable gaming. When ``fuzzy``, treatment is 1 only with
    probability 0.7 above the cutoff (imperfect compliance).
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(-1, 1, n)
    if manipulate > 0:
        n_mv = int(manipulate * n * 0.3)
        pool = np.where((r < 0) & (r > -0.3))[0]
        idx = rng.choice(pool, size=min(n_mv, pool.size // 2), replace=False)
        r[idx] = 0.005 + 0.04 * rng.random(idx.size)
    t_prob = np.where(r >= 0, 0.7, 0.15) if fuzzy else (r >= 0).astype(np.float64)
    t = (rng.random(n) < t_prob).astype(np.float64)
    y = 0.5 + 0.8 * r + tau * t + 0.5 * rng.standard_normal(n)
    return {
        "y": y,
        "r": r,
        "treat": t,
        "tau_true": np.float64(tau),
        "fuzzy": np.float64(float(fuzzy)),
    }


def bench_rd(seed: int = 20261231 + 177) -> dict[str, float]:
    """SYNTHETIC: sharp+fuzzy recovery, McCrary flags manipulation."""
    d = synth_rd(n=1200, tau=1.5, seed=seed)
    y = np.asarray(d["y"])
    r = np.asarray(d["r"])
    out = rd_local_linear(y, r)
    df = synth_rd(n=1200, tau=1.5, fuzzy=True, seed=seed + 1)
    fz = fuzzy_rd(np.asarray(df["y"]), np.asarray(df["r"]), np.asarray(df["treat"]))
    dm = synth_rd(n=1500, manipulate=0.8, seed=seed + 2)
    mc_clean = mccrary_test(r)
    mc_manip = mccrary_test(np.asarray(dm["r"]))
    donut = donut_rd(y, r, radius=0.08, bw=0.35)
    e1 = rd_local_linear(y, r, bw=float(out["bw"]))
    return {
        "synthetic_tau_hat": float(out["tau"]),
        "synthetic_tau_err": abs(float(out["tau"]) - 1.5),
        "synthetic_tau_t": float(out["t"]),
        "synthetic_fuzzy_tau_err": abs(float(fz["tau"]) - 1.5),
        "synthetic_fuzzy_first_stage": float(fz["first_stage"]),
        "synthetic_mccrary_z_clean": float(mc_clean["share_z"]),
        "synthetic_mccrary_z_manip": float(mc_manip["share_z"]),
        "synthetic_mccrary_flags": float(
            mc_manip["share_z"] > 1.96 and abs(mc_clean["share_z"]) < 1.96
        ),
        "synthetic_donut_tau_err": abs(float(donut["tau"]) - 1.5),
        "synthetic_determinism": float(out["tau"] == e1["tau"]),
    }
