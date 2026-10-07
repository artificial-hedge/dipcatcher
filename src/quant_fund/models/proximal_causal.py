"""Proximal causal inference via negative-control proxies.

References
----------
- Miao, W., Geng, Z. & Tchetgen Tchetgen, E.J. (2018). "Identifying
  Causal Effects with Proxy Variables of an Unmeasured Confounder."
  *Biometrika* 105(4), 987-993.
- Tchetgen Tchetgen, E.J., Ying, A., Cui, Y., Shi, X. & Miao, W. (2020).
  "An Introduction to Proximal Causal Learning." arXiv:2009.10982.
- Cui, Y., Pu, H., Shi, X., Miao, W. & Tchetgen Tchetgen, E.J. (2024).
  "Semiparametric Proximal Causal Inference." *JASA* 119(546), 1348-1359.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
When an unmeasured confounder U invalidates exchangeability, proximal
learning uses two proxies: a *treatment-inducing* proxy Z (Z indep of Y
given U, A, X) and an *outcome-inducing* proxy W (W indep of (A, Z)
given U, X). The outcome confounding bridge h(w, a, x) solves

    E[Y | Z, A=a, X=x] = int h(w, a, x) dF(w | z, a, x)

and E[Y(a)] = int h(w, a, x) dF(w, x) is then identified. Under a linear
bridge h = a0 + aw'w + aa*a + ax'x this is an instrumental-variable
problem: regress Y on (w, a, x) using (z, a, x) as instruments. The
synthetic generator satisfies the proxy conditional independences by
construction; the confounded naive estimate is visibly biased.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ols(y: FloatArray, x: FloatArray) -> FloatArray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def proximal_ate(
    y: FloatArray,
    treat: FloatArray,
    z_proxy: FloatArray,
    w_proxy: FloatArray,
    x: FloatArray | None = None,
) -> dict[str, float]:
    """2SLS proximal estimate of the ATE under unmeasured confounding.

    Instruments ``(z_proxy, treat, x)`` instrument ``(w_proxy, treat, x)``;
    the ATE is the fitted treatment coefficient plus the fitted
    treat-by-w interaction evaluated at the mean of w_proxy.
    """
    yy = np.asarray(y, dtype=np.float64)
    tt = np.asarray(treat, dtype=np.float64)
    zz = np.asarray(z_proxy, dtype=np.float64)
    ww = np.asarray(w_proxy, dtype=np.float64)
    if zz.ndim == 1:
        zz = zz[:, None]
    if ww.ndim == 1:
        ww = ww[:, None]
    if x is None:
        xx = np.zeros((yy.shape[0], 1), dtype=np.float64)
    else:
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx[:, None]
    n = yy.shape[0]
    if (
        yy.ndim != 1
        or tt.ndim != 1
        or tt.shape[0] != n
        or zz.shape[0] != n
        or ww.shape[0] != n
        or xx.shape[0] != n
    ):
        raise ValueError("y, treat, z_proxy, w_proxy, x must share n rows")
    if n < 60:
        raise ValueError("need n >= 60")
    if not all(np.all(np.isfinite(a)) for a in (yy, tt, zz, ww, xx)):
        raise ValueError("non-finite inputs")
    if not np.all((tt == 0.0) | (tt == 1.0)):
        raise ValueError("treat must be 0/1")

    one = np.ones((n, 1))
    # Outcome bridge h(w,a,x) = c + cw'w + ca*a + cx'x + caw*(a*w-sum).
    # Endogenous regressors: w and a*w (a exogenous, w proxied by z).
    aw = tt[:, None] * ww
    endo = np.column_stack([ww, aw])
    exo = np.column_stack([one, tt, xx])
    instr = np.column_stack([one, tt, xx, zz, tt[:, None] * zz])

    # First stage: each endogenous block on the instrument set.
    endo_hat = np.column_stack([instr @ _ols(endo[:, j], instr) for j in range(endo.shape[1])])
    reg2 = np.column_stack([exo, endo_hat])
    beta = _ols(yy, reg2)
    resid = yy - np.column_stack([exo, endo]) @ beta
    s2 = float(resid @ resid) / (n - reg2.shape[1])
    xtx_inv = np.linalg.pinv(reg2.T @ reg2)

    k_exo = exo.shape[1]
    k_w = ww.shape[1]
    ca = float(beta[1])
    caw = beta[k_exo + k_w : k_exo + 2 * k_w]
    wbar = ww.mean(axis=0)
    ate = float(ca + caw @ wbar)
    # ate = ca + caw'wbar — its SE needs the delta method over the full
    # coefficient block (gradient g' Sigma g), not just Var(ca): the old
    # form ignored Var(caw) and the covariance with ca.
    g = np.zeros(reg2.shape[1])
    g[1] = 1.0
    g[k_exo + k_w : k_exo + 2 * k_w] = wbar
    se_ate = float(np.sqrt(max(s2 * float(g @ xtx_inv @ g), 0.0)))

    # Naive coefficient for the honesty comparison.
    naive = float(_ols(yy, np.column_stack([one, tt, xx]))[1])
    # First-stage relevance: R^2 of the instrument block on w.
    f_w = float(np.mean(np.var(endo_hat[:, :k_w], axis=0) / np.var(endo[:, :k_w], axis=0)))
    return {
        "ate_proximal": ate,
        "se": se_ate,
        "naive_diff": naive,
        "first_stage_r2": f_w,
        "n_z": float(zz.shape[1]),
        "n_w": float(ww.shape[1]),
    }


def synth_proximal(
    n: int = 1500,
    seed: int = 20261231 + 277,
    effect: float = 1.0,
    confound: float = 0.8,
) -> dict[str, FloatArray]:
    """Two-proxy confounding synth.

    U ~ N(0,1); Z = U + ez (treatment-inducing proxy), W = 0.7U + ew
    (outcome-inducing). A ~ Bern(logit(-0.5 + confound*U)); Y = effect*A
    + confound*U + 0.5*X + ey. Naive regression recovers effect + confound
    bias; the proximal bridge isolates the effect.
    """
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    u = rng.normal(0.0, 1.0, n)
    x = rng.normal(0.0, 1.0, n)
    z = u + rng.normal(0.0, 0.5, n)
    w = 0.7 * u + rng.normal(0.0, 0.4, n)
    g = 1.0 / (1.0 + np.exp(-(-0.5 + confound * u + 0.3 * x)))
    a = (rng.uniform(0.0, 1.0, n) < g).astype(np.float64)
    y = effect * a + confound * u + 0.5 * x + rng.normal(0.0, 0.5, n)
    return {
        "y": y,
        "treat": a,
        "z": z,
        "w": w,
        "x": x,
        "true_ate": np.full(n, effect),
    }


def bench_proximal_causal(seed: int = 20261231 + 277) -> dict[str, float]:
    """Wave-48 self-check: proximal ATE close to truth where the naive
    difference is confounded."""
    d = synth_proximal(seed=seed)
    a = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    b = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    truth = 1.0
    detects = float(abs(a["ate_proximal"] - truth) < abs(a["naive_diff"] - truth) * 0.4)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == b),
        "synthetic_ate_proximal": a["ate_proximal"],
        "synthetic_naive_diff": a["naive_diff"],
        "synthetic_first_stage_r2": a["first_stage_r2"],
        "synthetic_se": a["se"],
    }
