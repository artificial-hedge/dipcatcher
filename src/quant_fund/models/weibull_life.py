"""Weibull life-data MLE on right-censored lifetimes.

Maximizes the censored log-likelihood over the shape beta via Newton
iterations (scale eta is profiled out in closed form:
eta^beta = sum t_i^beta / n_failed), then reports B10 life and the
fitted-vs-planted parameter errors on the shared synthetic fixture.
"""

import numpy as np

from quant_fund.models._rel_synth import WB_BETA, WB_CENSOR, WB_ETA, weibull_lifetimes


def _weibull_fit(t: np.ndarray, failed: np.ndarray) -> tuple[float, float]:
    beta = 1.0
    nf = float(np.sum(failed))
    for _ in range(60):
        tb = t**beta
        s0 = tb.sum()
        s_all = np.sum(tb * np.log(t))
        s_all2 = np.sum(tb * np.log(t) ** 2)
        dlog = np.sum(np.log(t[failed])) / nf
        g = 1.0 / beta + dlog - s_all / s0
        dg = -1.0 / beta**2 - (s_all2 * s0 - s_all**2) / s0**2
        step = g / dg
        beta -= float(np.clip(step, -0.5 * beta, 0.5 * beta))
        if abs(step) < 1e-10:
            break
    eta = float(((t**beta).sum() / nf) ** (1 / beta))
    return beta, eta


def bench_weibull_life(seed: int = 4901) -> dict[str, float]:
    t, failed = weibull_lifetimes(seed)
    beta, eta = _weibull_fit(t, failed)
    b10 = eta * (-np.log(0.90)) ** (1 / beta)
    b10_true = WB_ETA * (-np.log(0.90)) ** (1 / WB_BETA)
    return {
        "synthetic_wb_beta": beta,
        "synthetic_wb_beta_err": abs(beta - WB_BETA),
        "synthetic_wb_eta": eta,
        "synthetic_wb_eta_rel": abs(eta / WB_ETA - 1.0),
        "synthetic_wb_b10": b10,
        "synthetic_wb_b10_rel": abs(b10 / b10_true - 1.0),
        "synthetic_wb_frac_censored": float(1.0 - np.mean(failed)),
        "synthetic_wb_censor_at": float(WB_CENSOR),
    }
