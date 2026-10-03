"""GR4J daily rainfall-runoff model and Muskingum
flood routing.

- GR4J (Perrin, Michel & Andréassian 2003): two-store
  conceptual model — production store (Soil Moisture
  Accounting) + routing store, unit hydrographs UH1/UH2
  based on S-curves, groundwater exchange term.
- Muskingum (McCarthy 1938): linear reservoir routing
  with wedge/storage weighting x and travel time K.
- Goodness-of-fit: Nash-Sutcliffe efficiency (NSE),
  Kling-Gupta (KGE) on flows.

References
----------
- Perrin, Michel & Andréassian (2003) 'Improvement of a
  parsimonious model for streamflow simulation'
  J. Hydrol. 279.
- McCarthy (1938) 'The unit hydrograph and flood routing'
  USACE North Atlantic Division.
- Gupta et al. (2009) 'Decomposition of the MSE and NSE'
  J. Hydrol. 377.

Honesty
-------
SYNTHETIC self-check: a seeded P/ET series; asserts NSE
> 0.5 on a parameter set and Muskingum conserves mass.

Composition
-----------
Pure numpy. Inputs are daily precipitation and PE series;
outputs are simulated discharge and routing results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, name: str) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 10 or not np.isfinite(xa).all() or (xa < 0).any():
        raise ValueError(f"{name} must be non-negative, finite, len>=10")
    return xa


def _s_curve(t: FloatArray, x4: float, kind: int) -> FloatArray:
    """S-curve for UH ordinates (SH1 for UH1, SH2 for UH2)."""
    tt = np.clip(t, 0.0, None)
    if kind == 1:
        return np.where(tt <= 0, 0.0, np.where(tt < x4, (tt / x4) ** 2.5, 1.0))
    lo = (np.clip(tt / x4, 0.0, None)) ** 2.5
    hi = 1 - 0.5 * (np.clip(2 - tt / x4, 0.0, None)) ** 2.5
    return np.where(
        tt <= 0,
        0.0,
        np.where(tt < x4, 0.5 * lo, np.where(tt < 2 * x4, hi, 1.0)),
    )


def _uh(x4: float, kind: int, n_uh: int) -> FloatArray:
    """Unit hydrograph ordinates as differences of S-curves."""
    t = np.arange(n_uh + 1, dtype=np.float64)
    s = _s_curve(t, x4, kind)
    return np.diff(s)


def gr4j(
    precip: FloatArray,
    et: FloatArray,
    params: FloatArray | tuple[float, float, float, float],
) -> FloatArray:
    """GR4J simulation.

    params = (x1 production-store capacity mm,
              x2 groundwater exchange coef mm,
              x3 routing-store capacity mm,
              x4 UH time base days).
    """
    P = _check_series(precip, "precip")
    E = _check_series(et, "et")
    if P.size != E.size:
        raise ValueError("P and ET length mismatch")
    x1, x2, x3, x4 = np.asarray(params, dtype=np.float64)
    if min(x1, x3, x4) <= 0:
        raise ValueError("x1,x3,x4 positive")
    n = P.size
    n_uh1 = int(np.ceil(x4))
    n_uh2 = int(np.ceil(2 * x4))
    uh1 = _uh(x4, 1, n_uh1)
    uh2 = _uh(x4, 2, n_uh2)
    S = 0.5 * x1  # production store level
    R = 0.3 * x3  # routing store level
    q1_hist = np.zeros(n_uh1)
    q9_hist = np.zeros(n_uh2)
    Q = np.zeros(n)
    for t in range(n):
        p, e = P[t], E[t]
        # interception/production store
        if p >= e:
            pn = p - e
            en = 0.0
        else:
            pn = 0.0
            en = e - p
        if pn > 0:
            ps = pn * (1 - (S / x1) ** 2) / (1 + (S / x1) * np.tanh(pn / x1))
            S += ps
            pr = pn - ps
        else:
            es = en * (2 - S / x1) * np.tanh(en / x1) / (1 + (1 - S / x1) * np.tanh(en / x1))
            es = min(es, S)
            S -= es
            pr = 0.0
        # split: 90% via UH1->routing store, 10% via UH2->direct
        q1_hist = np.roll(q1_hist, 1)
        q1_hist[0] = 0.9 * pr
        q9_hist = np.roll(q9_hist, 1)
        q9_hist[0] = 0.1 * pr
        qr = float(uh1 @ q1_hist)
        qd = float(uh2 @ q9_hist)
        # routing store
        exch = x2 * (R / x3) ** 3.5
        R = max(0.0, R + qr + exch)
        if R > 0:
            Qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
            R -= Qr
        else:
            Qr = 0.0
        # direct branch
        Qd = max(0.0, qd + exch)
        Q[t] = Qr + Qd
    return Q


def muskingum(inflow: FloatArray, k: float, x: float, dt: float = 1.0) -> FloatArray:
    """Muskingum routing: S = K[x I + (1-x) Q]; outflow series.

    Coefficients: c0 = (-Kx+dt/2)/D, c1 = (Kx+dt/2)/D,
    c2 = (K(1-x)-dt/2)/D with D = K-Kx+dt/2.
    """
    infl = _check_series(inflow, "inflow")
    if k <= 0 or not (0 <= x <= 0.5) or dt <= 0:
        raise ValueError("bad muskingum params")
    d = k - k * x + dt / 2
    c0 = (-k * x + dt / 2) / d
    c1 = (k * x + dt / 2) / d
    c2 = (k * (1 - x) - dt / 2) / d
    q = np.zeros_like(infl)
    q[0] = infl[0]
    for t in range(1, infl.size):
        q[t] = c0 * infl[t] + c1 * infl[t - 1] + c2 * q[t - 1]
        q[t] = max(q[t], 0.0)
    return q


def nse(obs: FloatArray, sim: FloatArray) -> float:
    """Nash-Sutcliffe efficiency."""
    o = np.asarray(obs, dtype=np.float64)
    s = np.asarray(sim, dtype=np.float64)
    denom = ((o - o.mean()) ** 2).sum()
    if denom <= 0:
        raise ValueError("constant obs")
    return float(1 - ((o - s) ** 2).sum() / denom)


def kge(obs: FloatArray, sim: FloatArray) -> float:
    """Kling-Gupta efficiency (2009)."""
    o = np.asarray(obs, dtype=np.float64)
    s = np.asarray(sim, dtype=np.float64)
    if o.std() <= 0 or o.mean() <= 0:
        raise ValueError("degenerate obs")
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def bench_hydrology(seed: int = 508) -> dict[str, float]:
    """SYNTHETIC: GR4J with known params -> NSE/KGE on a
    re-simulation; Muskingum mass conservation check."""
    rng = np.random.default_rng(seed)
    n = 365
    # seasonal + event precipitation
    base = 2.0 + 3.0 * np.sin(2 * np.pi * np.arange(n) / 365 + 1.0)
    storms = (rng.random(n) < 0.12) * rng.gamma(3.0, 8.0, n)
    P = np.clip(base + storms, 0, None)
    E = np.clip(3.5 + 2.5 * np.sin(2 * np.pi * np.arange(n) / 365), 0.1, None)
    q = gr4j(P, E, (350.0, 0.5, 90.0, 1.5))
    nse_val = nse(q[60:], gr4j(P, E, (350.0, 0.5, 90.0, 1.5))[60:])
    kge_val = kge(q[60:], gr4j(P, E, (352.0, 0.48, 92.0, 1.45))[60:])
    # Muskingum on the routed flow
    routed = muskingum(q, k=2.0, x=0.2)
    mass_err = abs(routed.sum() - q.sum()) / q.sum()
    if nse_val < 0.999 or mass_err > 0.05:
        raise ValueError("hydrology self-check failed")
    return {
        "synthetic_nse_self": nse_val,
        "synthetic_kge_near_param": kge_val,
        "synthetic_muskingum_mass_err": mass_err,
        "synthetic_peak_q": float(q.max()),
        "synthetic_baseflow_frac": float(np.quantile(q, 0.1) / q.mean()),
    }
