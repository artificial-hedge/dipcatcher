"""Count-data regression: Negative Binomial, ZIP, hurdle, Vuong test.

Extends :mod:`quant_fund.models.discrete` (probit/logit/tobit) to the
integer-count family: Negative Binomial-2 (Cameron & Trivedi 1986)
with MLE of the overdispersion parameter, Lambert (1992) zero-inflated
Poisson, and the Mullahy (1986) hurdle two-part model. Vuong (1989)
non-nested LR test chooses between ZIP and plain Poisson; Pearson
chi-square diagnostics quantify overdispersion.

References
----------
- Cameron & Trivedi (1986). Econometric models based on count data.
  *J. Applied Econometrics* 1(1).
- Lambert (1992). Zero-inflated Poisson regression. *Technometrics* 34(1).
- Mullahy (1986). Specification and testing of some modified count
  data models. *J. Econometrics* 33(3).
- Vuong (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica* 57(2).

Honesty
-------
All panels are SYNTHETIC with planted overdispersion/zero-inflation.
Keys report parameter recovery, overdispersion-α recovery, Vuong
ordering on the right model class, and determinism — never claims
about real count processes (e.g. trade counts, event arrivals).

Composition notes
-----------------
- ``models/discrete.py``: binary/censored family — this module adds
  the count family on the same MLE-scoring machinery.
- ``models/gas_score.py`` (wave 30): GAS-Poisson time-varying
  intensity — this module is the static cross-sectional counterpart.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sopt
from scipy import stats as sstats
from scipy.special import gammaln

FloatArray = NDArray[np.float64]


def _check_count(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if ya.size != xa.shape[0] or ya.size < 20:
        raise ValueError("y and x must align with n>=20")
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(xa))):
        raise ValueError("inputs must be finite")
    if np.any(ya < 0) or np.any(np.abs(ya - np.round(ya)) > 1e-9):
        raise ValueError("y must be non-negative integer counts")
    if xa.shape[1] > 12:
        raise ValueError("too many regressors")
    return ya, np.column_stack([np.ones(xa.shape[0]), xa])


def _poisson_nll(beta: FloatArray, y: FloatArray, x: FloatArray) -> float:
    eta = np.clip(x @ beta, -20.0, 20.0)
    lam = np.exp(eta)
    return float(np.sum(lam - y * eta + gammaln(y + 1)))


def poisson_fit(y: FloatArray, x: FloatArray) -> dict[str, FloatArray | np.float64]:
    """Poisson GLM MLE + observed-information SEs + Pearson χ²."""
    ya, xa = _check_count(y, x)
    b0 = np.zeros(xa.shape[1])
    b0[0] = math.log(max(ya.mean(), 0.25))
    res = sopt.minimize(
        _poisson_nll,
        b0,
        args=(ya, xa),
        method="BFGS",
        options={"maxiter": 300},
    )
    beta = res.x
    eta = np.clip(xa @ beta, -20, 20)
    lam = np.exp(eta)
    w = lam
    info = (xa * w[:, None]).T @ xa
    se = np.sqrt(np.diag(np.linalg.inv(info + 1e-10 * np.eye(xa.shape[1]))))
    pearson = float(np.sum((ya - lam) ** 2 / np.maximum(lam, 1e-9)))
    pearson_p = float(sstats.chi2.sf(pearson, df=max(ya.size - xa.shape[1], 1)))
    return {
        "beta": np.asarray(beta),
        "se": np.asarray(se),
        "lambda_hat": np.asarray(lam),
        "pearson_chi2": np.float64(pearson),
        "pearson_p": np.float64(pearson_p),
        "nll": np.float64(res.fun),
    }


def negbin_fit(y: FloatArray, x: FloatArray) -> dict[str, FloatArray | np.float64]:
    """NB-2 MLE: mean λ = exp(xβ), variance λ(1 + α λ), α>=0.

    Joint optimization over (β, log α); reports the Pearson dispersion
    and a likelihood-ratio test of α=0 vs Poisson.
    """
    ya, xa = _check_count(y, x)
    k = xa.shape[1]

    def nll(theta: FloatArray) -> float:
        beta = theta[:k]
        loga = theta[k]
        alpha = math.exp(float(np.clip(loga, -20, 5)))
        eta = np.clip(xa @ beta, -20, 20)
        lam = np.exp(eta)
        r = 1.0 / alpha
        p_nb = r / (r + lam)
        ll = (
            gammaln(ya + r)
            - gammaln(r)
            - gammaln(ya + 1)
            + r * np.log(np.maximum(p_nb, 1e-300))
            + ya * np.log(np.maximum(1 - p_nb, 1e-300))
        )
        return -float(np.sum(ll))

    b0 = np.zeros(k + 1)
    b0[0] = math.log(max(ya.mean(), 0.25))
    b0[k] = -1.0
    res = sopt.minimize(nll, b0, method="BFGS", options={"maxiter": 400})
    beta = res.x[:k]
    alpha = float(math.exp(np.clip(res.x[k], -20, 5)))
    lam = np.exp(np.clip(xa @ beta, -20, 20))
    poi = poisson_fit(y, x)
    lr_alpha = float(2.0 * (float(poi["nll"]) - res.fun))
    p_alpha = float(
        0.5 * sstats.chi2.sf(max(lr_alpha, 0.0), df=1)
    )  # boundary → half-chi2 (Self-Liang)
    return {
        "beta": np.asarray(beta),
        "alpha": np.float64(alpha),
        "lambda_hat": np.asarray(lam),
        "nll": np.float64(res.fun),
        "lr_alpha_vs_poisson": np.float64(lr_alpha),
        "p_alpha": np.float64(p_alpha),
    }


def zip_fit(
    y: FloatArray, x: FloatArray, z_logit: FloatArray | None = None
) -> dict[str, FloatArray | np.float64]:
    """Zero-inflated Poisson: mixing π(z) via logit link (Lambert 1992).

    The zero-inflation design defaults to a constant π; pass ``z_logit``
    (n, q) design for covariate-dependent inflation.
    """
    ya, xa = _check_count(y, x)
    n = ya.size
    za = np.ones((n, 1)) if z_logit is None else np.asarray(z_logit)
    if za.ndim == 1:
        za = za[:, None]
    q = za.shape[1]
    k = xa.shape[1]

    def nll(theta: FloatArray) -> float:
        gamma = theta[:q]
        beta = theta[q:]
        pi = 1.0 / (1.0 + np.exp(-np.clip(za @ gamma, -20, 20)))
        eta = np.clip(xa @ beta, -20, 20)
        lam = np.exp(eta)
        ll = np.empty(n)
        is0 = ya == 0
        # P(Y=0) = π + (1-π) e^{-λ}
        ll[is0] = np.log(pi[is0] + (1 - pi[is0]) * np.exp(-lam[is0]) + 1e-300)
        ll[~is0] = np.log(1 - pi[~is0]) + sstats.poisson.logpmf(ya[~is0], lam[~is0])
        return -float(np.sum(ll))

    theta0 = np.zeros(q + k)
    theta0[q] = math.log(max(ya[ya > 0].mean(), 0.5))
    res = sopt.minimize(nll, theta0, method="BFGS", options={"maxiter": 400})
    gamma = res.x[:q]
    beta = res.x[q:]
    return {
        "gamma": np.asarray(gamma),
        "beta": np.asarray(beta),
        "pi_hat": np.asarray(1.0 / (1.0 + np.exp(-np.clip(za @ gamma, -20, 20)))),
        "nll": np.float64(res.fun),
    }


def hurdle_fit(y: FloatArray, x: FloatArray) -> dict[str, FloatArray | np.float64]:
    """Mullahy hurdle model: Bernoulli crossing + zero-truncated Poisson.

    Both parts share the design matrix here (parsimony); the crossing
    part is a plain logit on I(y>0), the count part a Poisson fit on
    the positive subsample.
    """
    ya, xa = _check_count(y, x)
    pos = ya > 0
    if pos.sum() < 10 or pos.sum() >= ya.size - 5:
        raise ValueError("need a nondegenerate zero/positive split")
    # logit on crossing
    from quant_fund.models.discrete import logit_fit  # existing family

    cross = logit_fit(pos.astype(np.float64), xa[:, 1:])
    cross_beta = np.asarray(cross["coef"])
    cross_fitted = np.asarray(cross["fitted"])
    # zero-truncated Poisson on positives via EM-free nll
    yp, xp = ya[pos], xa[pos]
    kp = xp.shape[1]

    def nll(beta: FloatArray) -> float:
        eta = np.clip(xp @ beta, -20, 20)
        lam = np.exp(eta)
        ll = sstats.poisson.logpmf(yp, lam) - np.log(np.maximum(1 - np.exp(-lam), 1e-300))
        return -float(np.sum(ll))

    b0 = np.zeros(kp)
    b0[0] = math.log(max(yp.mean(), 0.5))
    res = sopt.minimize(nll, b0, method="BFGS", options={"maxiter": 300})
    return {
        "gamma_cross": cross_beta,
        "beta_count": np.asarray(res.x),
        "p_cross_hat": cross_fitted,
        "nll_count": np.float64(res.fun),
        "share_pos": np.float64(float(pos.mean())),
    }


def vuong_zip_vs_poisson(y: FloatArray, x: FloatArray) -> dict[str, float]:
    """Vuong (1989) non-nested LR test: ZIP vs Poisson.

    Positive z favors ZIP; the statistic is the mean log-likelihood
    ratio divided by its standard error (asymptotically N(0,1) when
    the models are non-nested on the count outcome).
    """
    ya, xa = _check_count(y, x)
    poi = poisson_fit(y, x)
    zipm = zip_fit(y, x)
    lam_p = np.asarray(poi["lambda_hat"])
    lam_z = np.exp(np.clip(xa @ np.asarray(zipm["beta"]), -20, 20))
    pi = np.asarray(zipm["pi_hat"])
    # per-obs log-likelihoods
    ll_p = sstats.poisson.logpmf(ya, lam_p)
    ll_z = np.empty(ya.size)
    is0 = ya == 0
    ll_z[is0] = np.log(pi[is0] + (1 - pi[is0]) * np.exp(-lam_z[is0]))
    ll_z[~is0] = np.log(1 - pi[~is0]) + sstats.poisson.logpmf(ya[~is0], lam_z[~is0])
    d = ll_z - ll_p
    v = float(d.mean() / max(d.std(ddof=1), 1e-12) * math.sqrt(ya.size))
    return {
        "vuong_z": v,
        "vuong_p_zip_better": float(sstats.norm.sf(v)),
        "mean_lr": float(d.mean()),
    }


def synth_negbin(
    n: int = 400, alpha: float = 0.8, seed: int = 0
) -> dict[str, FloatArray | np.float64]:
    """Overdispersed counts: y ~ NB(λ(x), 1/α) with λ = exp(0.5 + 0.8x)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lam = np.exp(0.5 + 0.8 * x)
    r = 1.0 / alpha
    p_nb = r / (r + lam)
    y = rng.negative_binomial(r, p_nb)
    return {"y": np.asarray(y, dtype=np.float64), "x": x, "alpha_true": np.float64(alpha)}


def synth_zip(n: int = 500, pi: float = 0.35, seed: int = 0) -> dict[str, FloatArray | np.float64]:
    """ZIP: extra structural zeros mixed over Poisson(λ(x))."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lam = np.exp(0.4 + 0.6 * x)
    structural = rng.random(n) < pi
    y = np.where(structural, 0, rng.poisson(lam))
    return {"y": np.asarray(y, dtype=np.float64), "x": x, "pi_true": np.float64(pi)}


def bench_count_data(seed: int = 20261231 + 171) -> dict[str, float]:
    """SYNTHETIC count-model ordering: NB recovers α, Vuong picks ZIP."""
    nb = synth_negbin(n=500, alpha=0.8, seed=seed)
    nb_y = cast(FloatArray, nb["y"])
    nb_x = cast(FloatArray, nb["x"])
    fit_nb = negbin_fit(nb_y, nb_x)
    fit_p = poisson_fit(nb_y, nb_x)
    zp = synth_zip(n=600, pi=0.35, seed=seed + 1)
    zp_y = cast(FloatArray, zp["y"])
    zp_x = cast(FloatArray, zp["x"])
    fit_zip = zip_fit(zp_y, zp_x)
    v_test = vuong_zip_vs_poisson(zp_y, zp_x)
    beta_hat = np.asarray(fit_nb["beta"])
    beta_err = float(np.linalg.norm(beta_hat - np.array([0.5, 0.8])))
    zip_beta = np.asarray(fit_zip["beta"])
    zip_err = float(np.linalg.norm(zip_beta - np.array([0.4, 0.6])))
    d1 = negbin_fit(nb_y, nb_x)["alpha"]
    d2 = negbin_fit(nb_y, nb_x)["alpha"]
    return {
        "synthetic_nb_beta_err": beta_err,
        "synthetic_alpha_hat": float(fit_nb["alpha"]),
        "synthetic_alpha_err": float(abs(float(fit_nb["alpha"]) - 0.8)),
        "synthetic_lr_alpha_p": float(fit_nb["p_alpha"]),
        "synthetic_pearson_overdisp": float(fit_p["pearson_chi2"]) / 500.0,
        "synthetic_zip_beta_err": zip_err,
        "synthetic_pi_hat": float(np.asarray(fit_zip["pi_hat"]).mean()),
        "synthetic_vuong_z": float(v_test["vuong_z"]),
        "synthetic_vuong_picks_zip": float(v_test["vuong_z"] > 1.96),
        "synthetic_determinism": float(d1 == d2),
    }
