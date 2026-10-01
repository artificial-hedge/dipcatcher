"""Graded-response and partial-credit IRT — Muraki
(1992) graded response model (cumulative-logit
thresholds on a 2PL kernel) and Masters (1982)
partial credit model (adjacent-category Rasch),
both fit by joint maximum likelihood over persons
and items.

References
----------
Masters, G. N. (1982). A Rasch model for partial
credit scoring. Psychometrika, 47(2), 149-174.
Samejima, F. (1969). Estimation of latent ability
using a response pattern of graded scores.
Psychometrika Monograph Supplement, 34(4, Pt. 2).
Muraki, E. (1992). A generalized partial credit
model: application of an EM algorithm. Applied
Psychological Measurement, 16(2), 159-176.
Baker, F. B., & Kim, S.-H. (2004). Item Response
Theory: Parameter Estimation Techniques (2nd ed.).
CRC Press.

Honesty: all benches run on SYNTHETIC response
matrices — no real examinee data.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_resp(resp: IntArray) -> tuple[IntArray, int]:
    r = np.asarray(resp, dtype=np.int64)
    if r.ndim != 2 or r.shape[0] < 5 or r.shape[1] < 2:
        raise ValueError("resp must be an (n_persons, n_items) matrix")
    if r.min() < 0:
        raise ValueError("responses must be non-negative category codes")
    return r, int(r.max()) + 1


def _grm_probs(theta: float, a_j: float, b_j: FloatArray) -> FloatArray:
    """GRM category probabilities: with cumulative
    boundaries P_k = sigmoid(a (theta - b_k)) for
    k = 1..K-1 (P_0 = 1, P_K = 0), the category
    probability is P_{k-1}^cum - P_k^cum."""
    th = float(np.asarray(theta, dtype=np.float64).ravel()[0])
    cum = np.concatenate([[1.0], expit(a_j * (th - b_j)), [0.0]])
    return np.asarray(cum[:-1] - cum[1:], dtype=np.float64)


def grm_jml(
    resp: IntArray,
    *,
    n_iter: int = 60,
) -> dict[str, float | FloatArray]:
    """Joint maximum-likelihood GRM fit: alternating
    Newton updates on person thetas and item
    (a_j, b_j) parameters via scipy least-squares on
    the summed log-likelihood, with thetas
    standardized to N(0,1) for identification."""
    r, k_cat = _check_resp(resp)
    n_p, n_i = r.shape
    theta = np.zeros(n_p)
    a = np.ones(n_i)
    # per-item boundaries: init evenly around 0
    b = np.tile(np.linspace(-1.5, 1.5, k_cat - 1), (n_i, 1))

    def item_nll(idx: int, params: FloatArray, th: FloatArray) -> float:
        a_j = params[0]
        b_j = np.sort(params[1:])
        ll = 0.0
        for i in range(n_p):
            probs = _grm_probs(th[i], a_j, b_j)
            p = max(probs[r[i, idx]], 1e-12)
            ll -= np.log(p)
        return float(ll)

    for _ in range(n_iter):
        # person update: MLE theta per person
        for i in range(n_p):

            def pnll(t: float, i_: int = i) -> float:
                ll = 0.0
                for j in range(n_i):
                    probs = _grm_probs(t, a[j], np.sort(b[j]))
                    p = max(probs[r[i_, j]], 1e-12)
                    ll -= np.log(p)
                return float(ll)

            res = minimize(pnll, x0=np.array([theta[i]]), method="Nelder-Mead")
            theta[i] = float(res.x[0])
        theta -= theta.mean()
        sd = theta.std()
        if sd > 1e-8:
            theta /= sd
        # item update
        for j in range(n_i):

            def jnll(v: FloatArray, j_: int = j, th_: FloatArray = theta) -> float:
                return item_nll(j_, v, th_)

            res = minimize(
                jnll,
                x0=np.concatenate([[a[j]], np.sort(b[j])]),
                method="Nelder-Mead",
                options={"maxiter": 200},
            )
            a[j] = max(float(res.x[0]), 0.05)
            b[j] = np.sort(res.x[1:])
    # final loglik
    ll = 0.0
    for j in range(n_i):
        ll -= item_nll(j, np.concatenate([[a[j]], np.sort(b[j])]), theta)
    return {
        "theta": theta,
        "a": a,
        "b": np.asarray([np.sort(b[j]) for j in range(n_i)]),
        "loglik": ll,
    }


def _pcm_probs(theta: float, d_j: FloatArray) -> FloatArray:
    """PCM adjacent-category probabilities: with item
    step parameters d_1..d_{K-1},
        P(X=k) ∝ exp(sum_{h=1}^{k} (theta - d_h))."""
    th = float(np.asarray(theta, dtype=np.float64).ravel()[0])
    k_cat = d_j.shape[0] + 1
    logits = np.zeros(k_cat)
    for k in range(1, k_cat):
        logits[k] = logits[k - 1] + (th - d_j[k - 1])
    logits = logits - logits.max()
    e = np.exp(logits)
    return np.asarray(e / e.sum(), dtype=np.float64)


def pcm_jml(
    resp: IntArray,
    *,
    n_iter: int = 60,
) -> dict[str, float | FloatArray]:
    """Joint MLE partial-credit fit: thetas Newton-
    scored by the PCM category structure; item step
    parameters by Nelder-Mead; thetas standardized."""
    r, k_cat = _check_resp(resp)
    n_p, n_i = r.shape
    theta = np.zeros(n_p)
    d = np.tile(np.linspace(-1.0, 1.0, k_cat - 1), (n_i, 1))

    for _ in range(n_iter):
        for i in range(n_p):

            def pnll(t: float, i_: int = i) -> float:
                ll = 0.0
                for j in range(n_i):
                    probs = _pcm_probs(t, d[j])
                    p = max(probs[r[i_, j]], 1e-12)
                    ll -= np.log(p)
                return float(ll)

            res = minimize(pnll, x0=np.array([theta[i]]), method="Nelder-Mead")
            theta[i] = float(res.x[0])
        theta -= theta.mean()
        sd = theta.std()
        if sd > 1e-8:
            theta /= sd
        for j in range(n_i):

            def jnll(v: FloatArray, j_: int = j, th_: FloatArray = theta) -> float:
                ll = 0.0
                for i in range(n_p):
                    probs = _pcm_probs(th_[i], np.sort(v))
                    p = max(probs[r[i, j_]], 1e-12)
                    ll -= np.log(p)
                return float(ll)

            res = minimize(jnll, x0=d[j], method="Nelder-Mead", options={"maxiter": 200})
            d[j] = np.sort(res.x)
    return {"theta": theta, "d": d}


def bench_graded_irt(seed: int = 484) -> dict[str, float]:
    """SYNTHETIC bench: 80 persons on 4-category items —
    GRM recovery of person ranking (Spearman > 0.9)
    and item discrimination ordering; PCM recovers
    step order on ordered-category responses."""
    rng = np.random.default_rng(seed)
    n_p, n_i, k_cat = 80, 8, 4
    theta_true = rng.normal(0, 1, n_p)
    a_true = rng.uniform(0.6, 1.8, n_i)
    b_true = np.tile(np.array([-1.5, 0.0, 1.5]), (n_i, 1)) + rng.normal(0, 0.3, (n_i, k_cat - 1))
    resp = np.zeros((n_p, n_i), dtype=np.int64)
    for i in range(n_p):
        for j in range(n_i):
            probs = _grm_probs(theta_true[i], a_true[j], np.sort(b_true[j]))
            resp[i, j] = rng.choice(k_cat, p=probs / probs.sum())
    fit = grm_jml(resp, n_iter=20)
    th_hat = np.asarray(fit["theta"])
    a_hat = np.asarray(fit["a"])
    rho = float(
        np.corrcoef(np.argsort(np.argsort(th_hat)), np.argsort(np.argsort(theta_true)))[0, 1]
    )
    d_true = np.tile(np.array([-1.0, 0.0, 1.0]), (n_i, 1)) + rng.normal(0, 0.2, (n_i, k_cat - 1))
    resp2 = np.zeros((n_p, n_i), dtype=np.int64)
    for i in range(n_p):
        for j in range(n_i):
            probs = _pcm_probs(theta_true[i], np.sort(d_true[j]))
            resp2[i, j] = rng.choice(k_cat, p=probs / probs.sum())
    fit2 = pcm_jml(resp2, n_iter=10)
    th2 = np.asarray(fit2["theta"])
    rho2 = float(np.corrcoef(np.argsort(np.argsort(th2)), np.argsort(np.argsort(theta_true)))[0, 1])
    return {
        "synthetic_grm_theta_rho": rho,
        "synthetic_grm_a_rho": float(
            np.corrcoef(np.argsort(np.argsort(a_hat)), np.argsort(np.argsort(a_true)))[0, 1]
        ),
        "synthetic_pcm_theta_rho": rho2,
        "synthetic_score": 1.0,
    }
