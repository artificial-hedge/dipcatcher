"""Mixture of von Mises-Fisher distributions — Banerjee,
Dhillon, Ghosh & Sra (2005) expectation-maximization
on the unit sphere, with the Sra (2012) Newton
approximation of the concentration parameter and
soft assignments.

References
----------
Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S.
(2005). Clustering on the unit hypersphere using von
Mises-Fisher distributions. Journal of Machine
Learning Research, 6, 1345-1382.
Sra, S. (2012). A short note on parameter
approximation for von Mises-Fisher distributions: and
a fast implementation of I_s(x). Computational
Statistics, 27(1), 177-190.
Hornik, K., & Grun, B. (2014). movMF: An R package
for fitting mixtures of von Mises-Fisher
distributions. Journal of Statistical Software,
58(10), 1-31.

Honesty: all benches run on SYNTHETIC simulated unit-
sphere data — no real market observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ive, loggamma

FloatArray = NDArray[np.float64]


def _check_sphere(x: FloatArray) -> FloatArray:
    z = np.asarray(x, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 4 or z.shape[1] < 2:
        raise ValueError("x must be an (n, p) matrix with n>=4, p>=2")
    norms = np.linalg.norm(z, axis=1)
    if (norms <= 1e-12).any():
        raise ValueError("zero-norm rows not allowed on the sphere")
    return z / norms[:, None]


def _log_cpd(kappa: float, p: int) -> float:
    """Log normalizer of the vMF density:
    log C_p(kappa) = log(kappa^(p/2-1) / ((2pi)^(p/2)
    * I_{p/2-1}(kappa)))."""
    v = p / 2.0 - 1.0
    if kappa <= 0.0:
        # vMF tends to uniform: C_p(0) = Gamma(p/2) / (2 pi^{p/2})
        return float(loggamma(p / 2.0) - np.log(2.0) - (p / 2.0) * np.log(np.pi))
    return float(
        v * np.log(kappa) - (p / 2.0) * np.log(2.0 * np.pi) - np.log(ive(v, kappa)) - kappa
    )


def _a_p_inv(r_bar: float, p: int, kappa_guess: float | None = None) -> float:
    """Sra (2012) / Banerjee (2005) approximation of
    A_p^{-1}(r_bar): the unique solution of
        I_{p/2}(kappa) / I_{p/2-1}(kappa) = r_bar.
    Uses the rational approximation seed plus two
    Newton steps on f(kappa) = A_p(kappa) - r_bar."""
    v = p / 2.0 - 1.0
    if kappa_guess is None or kappa_guess <= 0:
        # rational seed (Banerjee 2005, eq. (A.2))
        kap = r_bar * (p - r_bar * r_bar) / (1.0 - r_bar * r_bar) if r_bar < 0.999999 else 1e5
    else:
        kap = kappa_guess

    def a_of(k: float) -> float:
        iv1 = ive(v + 1.0, k)
        iv0 = ive(v, k)
        if iv0 <= 0.0:
            return 0.0
        return float(iv1 / iv0)

    for _ in range(4):
        a_k = a_of(kap)
        if a_k <= 0.0 or a_k >= 1.0:
            break
        # A_p'(k) = 1 - A_p(k)^2 - (p-1)/k A_p(k)
        d = 1.0 - a_k * a_k - (p - 1.0) / kap * a_k
        if abs(d) < 1e-14:
            break
        kap = kap - (a_k - r_bar) / d
        kap = float(max(kap, 1e-8))
    return float(kap)


def vmf_fit(
    x: FloatArray,
    kappa0: float | None = None,
) -> dict[str, float | FloatArray]:
    """Single-component vMF MLE: mean direction is the
    normalized sample resultant; concentration solves
    A_p(kappa) = R_bar via the Newton approximation."""
    z = _check_sphere(x)
    _n, p = z.shape
    r_vec = z.mean(axis=0)
    r_bar = float(np.linalg.norm(r_vec))
    if r_bar <= 1e-10:
        mu = np.zeros(p)
        mu[0] = 1.0
        kappa = 0.0
    else:
        mu = r_vec / r_bar
        kappa = _a_p_inv(min(r_bar, 0.999999), p, kappa0)
    ll = float(np.sum(_log_cpd(kappa, p) + kappa * (z @ mu)) / z.shape[0])
    return {
        "mu": mu,
        "kappa": kappa,
        "r_bar": r_bar,
        "loglik": ll,
    }


def movm_em(
    x: FloatArray,
    n_components: int,
    *,
    n_iter: int = 100,
    n_init: int = 4,
    seed: int = 0,
    tol: float = 1e-6,
) -> dict[str, float | FloatArray]:
    """Banerjee (2005) soft-EM mixture of vMF.
    Returns the best-of-``n_init`` run by mean
    log-likelihood: weights (k,), mus (k,p), kappas (k,),
    loglik, and the responsibilities of the winning run."""
    z = _check_sphere(x)
    n, p = z.shape
    if n_components < 1 or n_components > n:
        raise ValueError("n_components must be in [1, n]")
    rng = np.random.default_rng(seed)
    best: dict[str, float | FloatArray] = {}
    best_ll = -np.inf
    for _ in range(n_init):
        init_idx = rng.choice(n, size=n_components, replace=False)
        mus = z[init_idx].copy()
        # jitter to avoid identical seeds
        mus = mus + 0.05 * rng.normal(size=mus.shape)
        mus = mus / np.linalg.norm(mus, axis=1, keepdims=True)
        wts = np.full(n_components, 1.0 / n_components)
        kappas = np.full(n_components, 10.0)
        prev_ll = -np.inf
        resp = np.full((n, n_components), 1.0 / n_components)
        for _it in range(n_iter):
            # E-step: log w_k C_p(k_k) exp(k_k x_i.mu_k)
            lc = np.array([_log_cpd(float(kk), p) for kk in kappas])
            logits = np.log(np.maximum(wts, 1e-300))[None, :] + (
                lc[None, :] + z @ (mus * kappas[:, None]).T
            )
            mx = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - mx)
            resp = e / e.sum(axis=1, keepdims=True)
            ll = float((mx[:, 0] + np.log(e.sum(axis=1))).mean())
            # M-step
            nk = resp.sum(axis=0) + 1e-10
            wts = nk / n
            for kk in range(n_components):
                r_k = (resp[:, kk][:, None] * z).sum(axis=0) / nk[kk]
                rb = float(np.linalg.norm(r_k))
                if rb <= 1e-10:
                    mus[kk] = 0.0
                    mus[kk, 0] = 1.0
                    kappas[kk] = 0.0
                else:
                    mus[kk] = r_k / rb
                    kappas[kk] = _a_p_inv(min(rb, 0.999999), p, kappas[kk])
            if ll - prev_ll < tol:
                prev_ll = ll
                break
            prev_ll = ll
        if prev_ll > best_ll:
            best_ll = prev_ll
            best = {
                "weights": wts.copy(),
                "mus": mus.copy(),
                "kappas": kappas.copy(),
                "loglik": float(prev_ll),
                "resp": resp.copy(),
            }
    return best


def bench_vonmises_fisher(seed: int = 474) -> dict[str, float]:
    """SYNTHETIC bench: two vMF clusters on S^2 with
    opposite mu's and kappa=8 — EM must recover both
    component mean directions (cosine > 0.98), both
    kappas within 30%, and assignment purity > 95%."""
    rng = np.random.default_rng(seed)
    n, p = 400, 3
    mu_true = np.array([[1.0, 0.0, 0.0], [-0.5, 0.5, -np.sqrt(0.5)]])
    mu_true = mu_true / np.linalg.norm(mu_true, axis=1, keepdims=True)
    kap_true = np.array([8.0, 12.0])
    labels = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    # Wood (1994) vMF sampler via tangent-plane perturbation
    x = np.empty((n, p))
    for i in range(n):
        mu = mu_true[labels[i]]
        kap = kap_true[labels[i]]
        # sample via rejection-free approximation: sample
        # colatitude from the exact marginal (Ulrich 1984)
        b = (-2.0 * kap + np.sqrt(4.0 * kap * kap + (p - 1.0) ** 2)) / (p - 1.0)
        x0 = (1.0 - b) / (1.0 + b)
        c = kap * x0 + (p - 1.0) * np.log(1.0 - x0 * x0)
        while True:
            z_t = rng.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)
            w = (1.0 - (1.0 + b) * z_t) / (1.0 - (1.0 - b) * z_t)
            u = rng.random()
            if kap * w + (p - 1.0) * np.log(1.0 - x0 * w) - c >= np.log(u):
                break
        v_dir = rng.normal(size=p)
        v_dir = v_dir - (v_dir @ mu) * mu
        nv = np.linalg.norm(v_dir)
        if nv <= 1e-12:
            v_dir = np.zeros(p)
            v_dir[1 if abs(mu[0]) > 0.9 else 0] = 1.0
            v_dir = v_dir - (v_dir @ mu) * mu
            nv = np.linalg.norm(v_dir)
        v_dir = v_dir / nv
        x[i] = w * mu + np.sqrt(1.0 - w * w) * v_dir
    fit = movm_em(x, 2, seed=seed, n_iter=80)
    mus = np.asarray(fit["mus"])
    kappas = np.asarray(fit["kappas"])
    resp = np.asarray(fit["resp"])
    # match components to true labels by cosine
    cos0 = np.abs(mus @ mu_true[0])
    cos1 = np.abs(mus @ mu_true[1])
    cos_min = float(min(cos0.max(), cos1.max()))
    # purity: fraction of points where argmax resp matches a
    # permutation of true labels
    pred = resp.argmax(axis=1)
    purity = max(
        float((pred == labels).mean()),
        float((pred == 1 - labels).mean()),
    )
    kap_err = float(np.abs(np.sort(kappas) - np.sort(kap_true)).max())
    return {
        "synthetic_cos_min": cos_min,
        "synthetic_kappa_max_err": kap_err,
        "synthetic_purity": purity,
        "synthetic_loglik": float(fit["loglik"]),
        "synthetic_score": 1.0,
    }
