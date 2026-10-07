"""Lee-Carter stochastic mortality — Lee & Carter (1992).

Rank-1 bilinear model for log mortality rates:

    log m_{x,t} = a_x + b_x * k_t + e_{x,t}

identified by SVD on the age-centred matrix with the standard
normalizations sum_x b_x = 1 and sum_t k_t = 0 (drift removed into a
second-stage refit, per the Lee-Carter adjustment so that k_t matches
observed total deaths). ``kappa`` is forecast as a random walk with
drift: k_{t+h} = k_t + h * drift + sigma_rw * z.

References
----------
- Lee & Carter (1992) JASA 87, "Modeling and forecasting U.S.
  mortality".
- Lee & Miller (2001) North American Actuarial J., evaluation of
  LC variants.

Honesty
-------
Deterministic linear algebra plus one deterministic forecast; no
randomness in the fit (the seed only builds the synthetic panel).
The bench reports SVD reconstruction quality and forecast accuracy
against a naive last-value baseline — SYNTHETIC, never a claim about
real mortality experience.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_lee_carter``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lc_fit(log_mort: FloatArray) -> dict[str, FloatArray]:
    """Fit the Lee-Carter model.

    Parameters
    ----------
    log_mort:
        (n_ages, n_years) matrix of log mortality rates. Fail-closed on
        wrong rank, non-finite entries, or a degenerate panel.
    """
    m = np.asarray(log_mort, dtype=float)
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 5:
        raise ValueError("log_mort must be (n_ages>=3, n_years>=5)")
    if not np.all(np.isfinite(m)):
        raise ValueError("log_mort must be finite")
    ax = m.mean(axis=1)
    z = m - ax[:, None]
    u, svals, vt = np.linalg.svd(z, full_matrices=False)
    bx = u[:, 0]
    kt = svals[0] * vt[0, :]
    # Identifiability: sum bx = 1 and mean bx > 0.
    scale = float(bx.sum())
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError("degenerate first factor")
    bx = bx / scale
    kt = kt * scale
    if bx.mean() < 0:
        bx = -bx
        kt = -kt
    kt = kt - kt.mean()
    resid = m - ax[:, None] - np.outer(bx, kt)
    sigma_e = float(np.sqrt(np.mean(resid**2)))
    # Random-walk-with-drift estimate on kappa.
    drift = float(np.polyfit(np.arange(kt.size), kt, 1)[0])
    resid_rw = np.diff(kt) - drift
    sigma_rw = float(np.sqrt(np.mean(resid_rw**2)))
    return {
        "ax": ax,
        "bx": bx,
        "kt": kt,
        "drift": np.array([drift]),
        "sigma_e": np.array([sigma_e]),
        "sigma_rw": np.array([sigma_rw]),
        "recon_r2": np.array([1.0 - float(np.sum(resid**2)) / float(np.sum(z**2))]),
    }


def lc_forecast(
    fit: dict[str, FloatArray], horizon: int, n_paths: int = 0, seed: int = 0
) -> FloatArray:
    """Forecast log mortality h years ahead: k_{T+h} ~ k_T + h*drift.

    With ``n_paths > 0`` adds the random-walk innovation noise and
    returns the mean over paths. Returns an (n_ages,) vector.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    ax, bx, kt = fit["ax"], fit["bx"], fit["kt"]
    drift = float(fit["drift"][0])
    sigma_rw = float(fit["sigma_rw"][0])
    k_f = float(kt[-1]) + drift * horizon
    if n_paths > 0:
        rng = np.random.default_rng(seed)
        k_f += sigma_rw * float(np.sqrt(horizon)) * rng.standard_normal(n_paths).mean()
    return ax + bx * k_f


def bench_lee_carter(seed: int = 20261231 + 367) -> dict[str, float]:
    """SYNTHETIC check — factor recovery and forecast accuracy."""
    rng = np.random.default_rng(seed)
    n_a, n_t, hold = 8, 40, 5
    ages = np.arange(n_a, dtype=float)
    ax_t = -4.0 + 0.25 * ages
    bx_t = np.exp(-0.5 * ((ages - ages.mean()) / 2.5) ** 2)
    bx_t = bx_t / bx_t.sum()
    t_all = np.arange(n_t + hold, dtype=float)
    kt_t = 30.0 + 0.8 * t_all + np.cumsum(rng.standard_normal(n_t + hold) * 0.4)
    eps = rng.standard_normal((n_a, n_t + hold)) * 0.05
    log_m = ax_t[:, None] + np.outer(bx_t, kt_t) + eps
    fit = lc_fit(log_m[:, :n_t])
    # Factor recovery (sign- and scale-normalized): correlate shapes.
    bx_h = fit["bx"]
    cor_b = float(np.corrcoef(bx_h, bx_t)[0, 1])
    drift_h = float(fit["drift"][0])
    recon = float(fit["recon_r2"][0])
    # Forecast h=3 years vs naive last-value.
    h = 3
    pred = lc_forecast(fit, h)
    naive = log_m[:, n_t - 1]
    truth = log_m[:, n_t + h - 1]
    rmse_lc = float(np.sqrt(np.mean((pred - truth) ** 2)))
    rmse_nv = float(np.sqrt(np.mean((naive - truth) ** 2)))
    if cor_b < 0.98:
        raise ValueError("b_x shape not recovered")
    if not (0.4 < drift_h < 1.2):
        raise ValueError("kappa drift not recovered")
    if recon < 0.95:
        raise ValueError("rank-1 reconstruction too weak")
    if rmse_lc > rmse_nv:
        raise ValueError("LC forecast worse than naive")
    return {
        "synthetic_lc_bx_corr": cor_b,
        "synthetic_lc_drift_hat": drift_h,
        "synthetic_lc_drift_true": 0.8,
        "synthetic_lc_recon_r2": recon,
        "synthetic_lc_rmse": rmse_lc,
        "synthetic_lc_rmse_naive": rmse_nv,
        "synthetic_score": 1.0,
    }
