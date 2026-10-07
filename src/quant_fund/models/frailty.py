"""Shared gamma frailty survival model (SYNTHETIC).

Unobserved heterogeneity across clusters: hazard h_ij(t) =
w_i · h0(t) · exp(xβ) with w_i ~ Gamma(1/θ, θ) iid per cluster.
The marginal likelihood integrates out the frailty (closed-form
Laplace for gamma), giving estimates of β and the dependence
parameter θ — ignored heterogeneity biases covariate effects
and understates duration dependence.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure parameter recovery on
generated clustered survival data — never market evidence.

References:
- Vaupel, J. W., Manton, K. G., Stallard, E. (1979). The impact
  of heterogeneity in individual frailty on the dynamics of
  mortality. *Demography* 16, 439-454.
- Clayton, D. (1978). A model for association in bivariate life
  tables. *Biometrika* 65, 141-151 — gamma-frailty copula.
- Nielsen, G. G., Gill, R. D., Andersen, P. K., Sørensen, T. I. A.
  (1992). A counting process approach to maximum likelihood
  estimation in frailty models. *Scand. J. Statistics* 19.
- Therneau, T. M., Grambsch, P. M. (2000). *Modeling Survival
  Data*, ch. 9 — penalized/EM frailty estimation.

Composition: pure numpy + scipy — Weibull baseline h0(t) =
λ p t^{p-1}, gamma-frailty marginal likelihood L_i =
(α+q-1)!/(α-1)! α^α (α+S_i)^{-(α+q)} ∏ h_ij^{δ}, profile MLE
on (β, log λ, log p, log θ); deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

FloatArray = NDArray[np.float64]


def frailty_fit(
    time: FloatArray,
    event: FloatArray,
    x: FloatArray,
    cluster: FloatArray,
) -> dict[str, float]:
    """Gamma shared-frailty Weibull MLE.

    ``cluster`` groups rows (each shares one frailty w_i).
    Returns β, baseline (λ, p), frailty variance θ, and the
    no-frailty reference for the LR contrast."""
    t = np.asarray(time, dtype=np.float64).ravel()
    d = np.asarray(event, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    cl = np.asarray(cluster).ravel()
    n = t.size
    if xx.ndim != 2 or xx.shape[0] != n:
        raise ValueError("x rows must match time")
    if d.size != n or cl.size != n:
        raise ValueError("event/cluster length mismatch")
    k = xx.shape[1]
    if n < 100 or k < 1 or k > 8:
        raise ValueError("n>=100, k in 1..8")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if np.any(t <= 0):
        raise ValueError("times must be positive")
    if not set(np.unique(d)).issubset({0.0, 1.0}):
        raise ValueError("event must be 0/1")
    if np.mean(d) < 0.1 or np.mean(d) > 0.95:
        raise ValueError("event share in .1..95")
    if np.min(np.std(xx, axis=0)) < 1e-9:
        raise ValueError("each covariate must vary")
    labels, inv = np.unique(cl, return_inverse=True)
    m = labels.size
    if m < 10:
        raise ValueError("need >=10 clusters")
    if m == n:
        raise ValueError("no shared frailty structure (1 obs/cluster)")

    idx = [np.where(inv == g)[0] for g in range(m)]

    def nll(theta: FloatArray) -> float:
        beta = theta[:k]
        lam = float(np.exp(np.clip(theta[k], -6, 3)))
        p = float(np.exp(np.clip(theta[k + 1], -2, 2)))
        alpha = float(np.exp(np.clip(theta[k + 2], -8, 4)))  # 1/θ
        xb = xx @ beta
        ll = 0.0
        for i in range(m):
            ii = idx[i]
            ti, di = t[ii], d[ii]
            si = np.exp(xb[ii])
            # cumulative baseline: Λ0 = λ t^p
            lam0 = lam * ti**p
            s_cum = float(np.sum(lam0 * si))
            q = float(np.sum(di))
            # marginal cluster ll: gammaln terms + hazards
            ll += float(gammaln(alpha + q) - gammaln(alpha))
            ll += alpha * np.log(alpha)
            ll -= (alpha + q) * np.log(alpha + s_cum)
            if q > 0:
                ll += float(np.sum(di * (xb[ii] + np.log(lam) + np.log(p) + (p - 1) * np.log(ti))))
        return -ll

    res = minimize(
        nll,
        np.zeros(k + 3),
        method="Nelder-Mead",
        options={"maxiter": 4000},
    )
    beta = res.x[:k]
    lam = float(np.exp(np.clip(res.x[k], -6, 3)))
    p = float(np.exp(np.clip(res.x[k + 1], -2, 2)))
    theta_var = float(np.exp(-np.clip(res.x[k + 2], -8, 4)))

    # no-frailty Weibull reference (θ=0, α→∞)
    def nll0(theta0: FloatArray) -> float:
        beta = theta0[:k]
        lam = float(np.exp(np.clip(theta0[k], -6, 3)))
        p = float(np.exp(np.clip(theta0[k + 1], -2, 2)))
        xb = xx @ beta
        lam0 = lam * t**p
        ll = float(np.sum(-lam0 * np.exp(xb)))
        ll += float(np.sum(d * (xb + np.log(lam) + np.log(p) + (p - 1) * np.log(t))))
        return -ll

    res0 = minimize(nll0, np.zeros(k + 2), method="Nelder-Mead", options={"maxiter": 3000})
    beta0 = res0.x[:k]
    ll_frailty = -float(res.fun)
    ll_null = -float(res0.fun)

    return {
        "n": float(n),
        "m_clusters": float(m),
        "beta_x": float(beta[0]),
        "beta_x_nofrailty": float(beta0[0]),
        "lambda": lam,
        "p": p,
        "theta": theta_var,
        "ll_frailty": ll_frailty,
        "ll_null": ll_null,
        "lr_frailty": float(2 * (ll_frailty - ll_null)),
        "event_share": float(np.mean(d)),
    }


def synth_frailty(
    n_clusters: int = 40,
    per_cluster: int = 20,
    beta_x: float = 0.7,
    theta: float = 0.8,
    censor: float = 0.25,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Weibull(λ=0.3, p=1.3) baseline · gamma(1/θ) cluster frailty,
    administrative censoring."""
    rng = np.random.default_rng(seed)
    m, pc = n_clusters, per_cluster
    lam0, p0 = 0.3, 1.3
    w = rng.gamma(1.0 / theta, theta, m)
    x = rng.normal(0.0, 1.0, (m * pc, 1))
    cl = np.repeat(np.arange(m), pc)
    xb = beta_x * x[:, 0]
    u = rng.uniform(0.0, 1.0, m * pc)
    # invert weibull CDF with frailty: t = (-ln u / (λ w e^{xβ}))^{1/p}
    t_event = (-np.log(u) / (lam0 * w[cl] * np.exp(xb))) ** (1.0 / p0)
    t_cens = np.quantile(t_event, 1.0 - censor * 0.9)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    return {
        "time": time,
        "event": event,
        "x": x,
        "cluster": cl.astype(np.float64),
        "beta_x": np.array([beta_x]),
        "theta": np.array([theta]),
    }


def bench_frailty(seed: int = 20261231 + 233) -> dict[str, float]:
    """Frailty self-check: θ and β recovered under clustering;
    no-frailty fit attenuates β. All ``synthetic_*``."""
    d = synth_frailty(theta=0.8, seed=seed)
    out = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    d0 = synth_frailty(theta=0.01, seed=seed + 1)
    out0 = frailty_fit(
        np.asarray(d0["time"]),
        np.asarray(d0["event"]),
        np.asarray(d0["x"]),
        np.asarray(d0["cluster"]),
    )
    out_b = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )

    th = float(out["theta"])
    b = float(out["beta_x"])
    return {
        "synthetic_theta": th,
        "synthetic_theta_err": float(abs(th - 0.8)),
        "synthetic_beta": b,
        "synthetic_beta_err": float(abs(b - 0.7)),
        "synthetic_beta_nofrailty": float(out["beta_x_nofrailty"]),
        "synthetic_lr_frailty": float(out["lr_frailty"]),
        "synthetic_theta_homog": float(out0["theta"]),
        "synthetic_detects": float(
            th > 0.3
            and abs(b - 0.7) < 0.25
            and th > float(out0["theta"]) * 1.5
            and float(out["lr_frailty"]) > 5.0
        ),
        "synthetic_determinism": float(th == float(out_b["theta"]) and b == float(out_b["beta_x"])),
    }
