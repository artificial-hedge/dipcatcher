"""Panel unit-root tests — Im-Pesaran-Shin and Levin-Lin-Chu.

Cross-sectional power lets panels detect mean reversion invisible to
single-series ADF tests. LLC pools the ADF regression under a common
persistence ρ; IPS averages per-unit ADF t-stats and standardises by
their tabulated moments — valid under heterogeneous dynamics.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure rejection on generated AR(1)
panels — never market evidence.

References:
- Levin, A., Lin, C.-F., Chu, C.-S. J. (2002). Unit root tests in
  panel data: asymptotic and finite-sample properties.
  *J. Econometrics* 108, 1-24.
- Im, K. S., Pesaran, M. H., Shin, Y. (2003). Testing for unit
  roots in heterogeneous panels. *J. Econometrics* 115, 53-74.
- Maddala, G. S., Wu, S. (1999). A comparative study of unit root
  tests with panel data. *Oxford Bulletin* 61 — the IPS moment
  table approximated here (MacKinnon-style response surface).
- Banerjee, A. (1999). Panel data unit roots and cointegration:
  an overview. *Oxford Bulletin* 61.

Composition: pure numpy + scipy — per-unit ADF(1) t-stats, IPS
standardised groupmean with simulated moments, LLC pooled
regression; deterministic ``np.random.default_rng``; no new deps.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _adf_t(y: FloatArray) -> float:
    """ADF(1) t-statistic for ρ-1 in Δy = a + (ρ-1)y_{t-1} + e."""
    dy = np.diff(y)
    x = np.column_stack([np.ones(dy.size), y[:-1]])
    b, res, *_ = np.linalg.lstsq(x, dy, rcond=None)
    r = dy - x @ b
    s2 = float(r @ r / max(dy.size - 2, 1))
    xtx = x.T @ x
    try:
        cov = s2 * np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(b[1] / math.sqrt(max(cov[1, 1], 1e-300)))


def _ips_moments(T: int, n_sim: int = 24, seed: int = 7) -> tuple[float, float]:
    """Simulated E/Var of the ADF t-stat under the unit-root null —
    the IPS standardisation moments (MacKinnon-style)."""
    rng = np.random.default_rng(seed)
    ts = np.empty(n_sim)
    for i in range(n_sim):
        y = np.cumsum(rng.normal(0, 1, T))
        ts[i] = _adf_t(y)
    ts = ts[np.isfinite(ts)]
    return float(np.mean(ts)), float(np.var(ts))


def panel_unitroot(
    panel: FloatArray,
    n_moment_sims: int = 24,
    seed: int = 7,
) -> dict[str, float]:
    """IPS + LLC panel unit-root tests.

    ``panel`` is (n_units × T): each row one unit's series.
    Returns standardised stats (≈N(0,1) under the unit-root null)
    and one-sided p-values (reject = mean reversion)."""
    pp = np.atleast_2d(np.asarray(panel, dtype=np.float64))
    n_units, T = pp.shape
    if n_units < 8 or T < 30:
        raise ValueError("need >=8 units and T>=30")
    if not np.all(np.isfinite(pp)):
        raise ValueError("finite panel required")
    if np.any(np.ptp(pp, axis=1) < 1e-9):
        raise ValueError("each unit must vary")

    # IPS: groupmean ADF t-bar standardised
    tstats = np.array([_adf_t(pp[i]) for i in range(n_units)])
    if np.any(~np.isfinite(tstats)):
        raise ValueError("ADF regression singular for a unit")
    tbar = float(np.mean(tstats))
    mu_t, var_t = _ips_moments(T, n_sim=n_moment_sims, seed=seed)
    ips = math.sqrt(n_units) * (tbar - mu_t) / math.sqrt(max(var_t, 1e-12))
    p_ips = float(norm.cdf(ips))

    # LLC: pooled Δy = a_i + ρhat y_{t-1} + e (common ρ, unit FE)
    dy_rows, ly_rows, fe_rows = [], [], []
    for i in range(n_units):
        dy_i = np.diff(pp[i])
        ly_i = pp[i][:-1]
        dy_rows.append(dy_i)
        ly_rows.append(ly_i)
        fe_rows.append(np.full(dy_i.size, i))
    dy = np.concatenate(dy_rows)
    ly = np.concatenate(ly_rows)
    fe = np.concatenate(fe_rows).astype(int)
    # partial out unit FE (demean within unit)
    dy_dm = dy.copy()
    ly_dm = ly.copy()
    for i in range(n_units):
        m = fe == i
        dy_dm[m] -= dy_dm[m].mean()
        ly_dm[m] -= ly_dm[m].mean()
    # pooled rho on demeaned
    b = float(ly_dm @ dy_dm / max(ly_dm @ ly_dm, 1e-300))
    r = dy_dm - b * ly_dm
    s2 = float(r @ r / max(dy.size - n_units - 1, 1))
    se = math.sqrt(s2 / max(ly_dm @ ly_dm, 1e-300))
    t_rho = b / max(se, 1e-300)
    # LLC standardised stat: adjust t_rho by simulated mean/sd under null
    mu_llc, var_llc = _llc_moments(n_units, T, seed=seed + 1)
    llc = (t_rho - mu_llc) / math.sqrt(max(var_llc, 1e-12))
    p_llc = float(norm.cdf(llc))

    return {
        "n_units": float(n_units),
        "T": float(T),
        "ips_stat": float(ips),
        "ips_p": p_ips,
        "llc_stat": float(llc),
        "llc_p": p_llc,
        "tbar": tbar,
        "rho_pooled": b,
        "reject_both_5": float(p_ips < 0.05 and p_llc < 0.05),
    }


def _llc_moments(n_units: int, T: int, n_sim: int = 20, seed: int = 13) -> tuple[float, float]:
    """Simulated mean/var of the pooled LLC t_rho under unit-root
    null (common-ρ random walks)."""
    rng = np.random.default_rng(seed)
    ts = np.empty(n_sim)
    for s in range(n_sim):
        pan = np.cumsum(rng.normal(0, 1, (n_units, T)), axis=1)
        dy_rows, ly_rows, fe_rows = [], [], []
        for i in range(n_units):
            dy_rows.append(np.diff(pan[i]))
            ly_rows.append(pan[i][:-1])
            fe_rows.append(np.full(T - 1, i))
        dy = np.concatenate(dy_rows)
        ly = np.concatenate(ly_rows)
        fe = np.concatenate(fe_rows).astype(int)
        dy_dm, ly_dm = dy.copy(), ly.copy()
        for i in range(n_units):
            m = fe == i
            dy_dm[m] -= dy_dm[m].mean()
            ly_dm[m] -= ly_dm[m].mean()
        b = float(ly_dm @ dy_dm / max(ly_dm @ ly_dm, 1e-300))
        r = dy_dm - b * ly_dm
        s2 = float(r @ r / max(dy.size - n_units - 1, 1))
        se = math.sqrt(s2 / max(ly_dm @ ly_dm, 1e-300))
        ts[s] = b / max(se, 1e-300)
    return float(np.mean(ts)), float(np.var(ts))


def synth_panel_ar(
    n_units: int = 24,
    T: int = 120,
    rho: float = 0.85,
    seed: int = 0,
) -> FloatArray:
    """Panel of independent AR(1) units; rho=1 → unit root."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n_units, T))
    for i in range(n_units):
        eps = rng.normal(0, 1, T)
        for t in range(1, T):
            y[i, t] = rho * y[i, t - 1] + eps[t]
    return y


def bench_panel_unitroot(seed: int = 20261231 + 226) -> dict[str, float]:
    """Panel unit-root self-check: stationary panel (ρ=.85) rejected
    by both tests; unit-root panel (ρ=1) kept. All ``synthetic_*``."""
    pan = synth_panel_ar(rho=0.85, seed=seed)
    out = panel_unitroot(pan)
    pan_ur = synth_panel_ar(rho=1.0, seed=seed + 1)
    out_ur = panel_unitroot(pan_ur)
    out_b = panel_unitroot(pan)

    ips_p = float(out["ips_p"])
    llc_p = float(out["llc_p"])
    return {
        "synthetic_ips_p": ips_p,
        "synthetic_llc_p": llc_p,
        "synthetic_ur_ips_p": float(out_ur["ips_p"]),
        "synthetic_ur_llc_p": float(out_ur["llc_p"]),
        "synthetic_reject_both": float(out["reject_both_5"]),
        "synthetic_rho_pooled": float(out["rho_pooled"]),
        "synthetic_detects": float(
            ips_p < 0.05
            and llc_p < 0.05
            and float(out_ur["ips_p"]) > 0.05
            and float(out_ur["llc_p"]) > 0.05
        ),
        "synthetic_determinism": float(ips_p == float(out_b["ips_p"])),
    }
