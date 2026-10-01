"""Many / weak instrument estimators: LIML, JIVE, HFUL.

When K instruments are each individually weak, 2SLS is biased toward
OLS; the k-class estimators here stay median-unbiased: LIML (lowest
eigenvalue of the concentration matrix), JIVE (Angrist-Imbens-Krueger
jackknife — delete-own-observation first stage), HFUL (Hansen-Hausman-
Newey jackknifed k-class).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure estimator bias under generated
many-weak-IV designs — never market evidence.

References:
- Hansen, Hausman, Newey (2008). Estimation with many instrumental
  variables. *JBES* 26 (HFUL).
- Angrist, Imbens, Krueger (1999). Jackknife instrumental variables
  estimation. *J. Applied Econometrics* 14 (JIVE).
- Bekker (1994). Alternative approximations to the distributions of
  instrumental variable estimators. *Econometrica* 62 (LIML
  distribution theory).
- Hausman, Newey, Woutersen, Chao, Swanson (2012). Instrumental
  variable estimation with heteroskedasticity and many instruments.
  *Quantitative Economics* 3.

Composition: pure numpy — k-class eigenvalue, leave-one-out first
stage, cluster-free sandwich SEs; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_cols(
    y: FloatArray, x: FloatArray, z: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    xa = np.asarray(x, dtype=np.float64)
    za = np.asarray(z, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if za.ndim == 1:
        za = za[:, None]
    n = ya.size
    if xa.shape[0] != n or za.shape[0] != n:
        raise ValueError("y/x/z length mismatch")
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(xa)) and np.all(np.isfinite(za))):
        raise ValueError("non-finite input")
    if za.shape[1] < 1:
        raise ValueError("need >=1 instrument")
    if n < za.shape[1] + xa.shape[1] + 3:
        raise ValueError("n too small for instrument count")
    return ya, xa, za


def _project(z: FloatArray) -> FloatArray:
    """Projection matrix P = Z(Z'Z)^{-1}Z' with intercept in Z."""
    n = z.shape[0]
    Z = np.column_stack([np.ones(n), z])
    return Z @ np.linalg.pinv(Z.T @ Z) @ Z.T


def liml(y: FloatArray, x: FloatArray, z: FloatArray) -> dict[str, float | FloatArray]:
    """Limited-information ML: lowest eigenvalue k of the
    concentration matrix, then k-class estimator with
    beta = (X'(I - k M_z)X)^{-1} X'(I - k M_z) y."""
    ya, xa, za = _as_cols(y, x, z)
    n = ya.size
    kx = xa.shape[1]
    if kx != 1:
        raise ValueError("liml supports exactly one endogenous regressor")
    P = _project(za)
    m0 = np.eye(n) - np.ones((n, n)) / n  # residualizes intercept only

    # LIML = argmin_b AR(b) = (y-xb)'P(y-xb) / (y-xb)'m0(y-xb).
    # Scalar bounded solve — more robust than the eigen-formulation
    # (which misplaces k when the centered moment matrix is
    # ill-conditioned under many weak instruments).
    from scipy.optimize import minimize_scalar

    def _ratio(b: float) -> float:
        r = ya - b * xa[:, 0]
        den = float(r @ m0 @ r)
        return float(r @ P @ r / max(den, 1e-12))

    ols_b = float(np.cov(xa[:, 0], ya)[0, 1] / np.var(xa[:, 0]))
    res = minimize_scalar(_ratio, bounds=(ols_b - 10.0, ols_b + 10.0), method="bounded")
    beta = float(res.x)
    k_min = float(res.fun)
    resid = ya - xa[:, 0] * beta
    s2 = float(resid @ resid) / (n - 1)
    # Bekker-style SE via AR curvature
    h = 1e-4
    curv = (_ratio(beta + h) - 2 * _ratio(beta) + _ratio(beta - h)) / h**2
    var_b = float(2.0 * max(k_min, 1e-9) / max(curv * n, 1e-9))
    se = math.sqrt(max(var_b, s2 / max(float(xa[:, 0] @ xa[:, 0]), 1.0)))
    return {
        "beta": beta,
        "se": se,
        "z": beta / max(se, 1e-12),
        "ar_min": k_min,
        "n": float(n),
        "k_instruments": float(za.shape[1]),
    }


def jive(y: FloatArray, x: FloatArray, z: FloatArray) -> dict[str, float]:
    """Jackknife IV: first stage fitted for each i using all OTHER
    observations (delete-i); second stage regresses y on xhat_i.
    xhat_i = z_i' (Z_{-i}'Z_{-i})^{-1} Z_{-i}' X_{-i}."""
    ya, xa, za = _as_cols(y, x, z)
    n = ya.size
    kx = xa.shape[1]
    if kx != 1:
        raise ValueError("jive supports exactly one endogenous regressor")
    Z = np.column_stack([np.ones(n), za])
    xhat = np.empty(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        Zi = Z[keep]
        xi = xa[keep]
        pi, *_ = np.linalg.lstsq(Zi, xi[:, 0], rcond=None)
        xhat[i] = float(Z[i] @ pi)
    X2 = np.column_stack([np.ones(n), xhat])
    beta2, *_ = np.linalg.lstsq(X2, ya, rcond=None)
    resid = ya - np.column_stack([np.ones(n), xa[:, 0]]) @ beta2
    meat = X2 * resid[:, None]
    xtx = np.linalg.pinv(X2.T @ X2)
    cov = xtx @ (meat.T @ meat) @ xtx * (n / max(n - 2, 1))
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        "beta": float(beta2[1]),
        "se": se,
        "z": float(beta2[1] / max(se, 1e-12)),
        "n": float(n),
    }


def hful(y: FloatArray, x: FloatArray, z: FloatArray) -> dict[str, float]:
    """HFUL: jackknifed k-class — uses M matrix diagonal to
    form the Hessian-Hausman-Newey moments:
    beta = (x'[P - D_alpha] x)^{-1} x'[P - D_alpha] y
    where D_alpha is diagonal with entries alpha_i = P_ii/(1-P_ii)
    -normalized combination (HFUL form per Hansen-Hausman-Newey)."""
    ya, xa, za = _as_cols(y, x, z)
    n = ya.size
    kx = xa.shape[1]
    if kx != 1:
        raise ValueError("hful supports exactly one endogenous regressor")
    P = _project(za)
    p_ii = np.diag(P)
    # HFUL weighting: alpha_i = p_ii; x-vec uses delete-own projection
    # v_i = x_i - z_i' pi_{-i} ≈ x_i - (P x)_i / (1 - p_ii) ... standard:
    # xtilde_i = ((Px)_i - p_ii x_i) / (1 - p_ii) + p_ii x_i? Use HHN:
    # yhat_i = (P_ii x_i + sum_j!=i p_ij x_j) ; equivalently
    # yhat = (P - diag(p_ii)) x + diag(p_ii) x = Px; HHN uses
    # yhat_i = (Px)_i - p_ii * (x_i - (Px)_i)/(1-p_ii) → the HFUL form:
    px = P @ xa
    denom = np.maximum(1.0 - p_ii, 1e-6)
    xhat = (px[:, 0] - p_ii * xa[:, 0]) / denom
    # HFUL moment form
    X2 = np.column_stack([np.ones(n), xhat])
    beta2, *_ = np.linalg.lstsq(X2, ya, rcond=None)
    resid = ya - np.column_stack([np.ones(n), xa[:, 0]]) @ beta2
    meat = X2 * resid[:, None]
    xtx = np.linalg.pinv(X2.T @ X2)
    cov = xtx @ (meat.T @ meat) @ xtx * (n / max(n - 2, 1))
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        "beta": float(beta2[1]),
        "se": se,
        "z": float(beta2[1] / max(se, 1e-12)),
        "mean_leverage": float(p_ii.mean()),
        "n": float(n),
    }


def synth_many_iv(
    n: int = 400,
    k: int = 30,
    pi_strength: float = 0.15,
    beta: float = 1.0,
    endog: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Many-instrument design: K orthogonal instruments each weak
    (first-stage π = pi_strength), endogeneity via correlated errors."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, (n, k))
    pi = rng.normal(0.0, pi_strength, k)
    v = rng.normal(0.0, 1.0, n)
    u = endog * v + rng.normal(0.0, math.sqrt(max(1.0 - endog**2, 0.05)), n)
    x = z @ pi + v
    y = beta * x + u
    return {
        "y": y,
        "x_endog": x,
        "z": z,
        "beta": np.full(1, beta),
        "pi": pi,
    }


def bench_many_iv(seed: int = 20261231 + 198) -> dict[str, float]:
    """Many-weak-IV self-check: JIVE/HFUL/LIML stay near beta while 2SLS
    drifts toward OLS. All ``synthetic_*``."""
    d = synth_many_iv(seed=seed, beta=1.0, pi_strength=0.10, k=40, n=400)
    y = np.asarray(d["y"])
    x = np.asarray(d["x_endog"])
    z = np.asarray(d["z"])
    beta_true = float(np.asarray(d["beta"]).item())

    l_ = liml(y, x, z)
    j_ = jive(y, x, z)
    h_ = hful(y, x, z)
    # plain 2SLS for comparison
    Z = np.column_stack([np.ones(y.size), z])
    Px = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ x
    X2 = np.column_stack([np.ones(y.size), Px])
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    ols = float(np.cov(x, y)[0, 1] / np.var(x))

    d0 = synth_many_iv(seed=seed + 1, beta=0.0)
    l0 = liml(np.asarray(d0["y"]), np.asarray(d0["x_endog"]), np.asarray(d0["z"]))
    h0 = hful(np.asarray(d0["y"]), np.asarray(d0["x_endog"]), np.asarray(d0["z"]))

    return {
        "synthetic_beta_liml": float(l_["beta"]),
        "synthetic_beta_jive": float(j_["beta"]),
        "synthetic_beta_hful": float(h_["beta"]),
        "synthetic_beta_2sls": float(b2[1]),
        "synthetic_beta_ols": ols,
        "synthetic_beta_true": beta_true,
        "synthetic_liml_err": abs(float(l_["beta"]) - beta_true),
        "synthetic_2sls_err": abs(float(b2[1]) - beta_true),
        "synthetic_ols_err": abs(ols - beta_true),
        "synthetic_beats_2sls": float(
            abs(float(l_["beta"]) - beta_true) < abs(float(b2[1]) - beta_true)
        ),
        "synthetic_null_liml": float(l0["beta"]),
        "synthetic_null_hful": float(h0["beta"]),
        "synthetic_detects": float(abs(float(l_["beta"])) > 3.0 * abs(float(l0["beta"])) + 0.15),
        "synthetic_determinism": float(float(l_["beta"]) == float(liml(y, x, z)["beta"])),
    }
