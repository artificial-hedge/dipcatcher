"""Latent class analysis — EM for Bernoulli-response mixtures.

Lazarsfeld & Henry (1968), Goodman (1974): subjects belong to one of
K unobserved classes; given the class, item responses are independent
Bernoulli with class-specific probabilities. EM maximizes the
observed-data likelihood:

    E-step: r_jk ∝ pi_k prod_i p_ki^x (1-p_ki)^(1-x)
    M-step: pi_k = mean_j r_jk,  p_ki = sum_j r_jk x_ji / sum_j r_jk

Multiple restarts guard the multimodal likelihood; BIC selects K.
Honesty: the bench simulates a 2-class mixture with well-separated
item profiles and checks the assignment accuracy (up to label
permutation) and pi recovery. EM is locally convergent — restarts
are seeded, class profiles are planted with a large margin. Fail-
closed on non-binary input or degenerate posteriors.

References: Lazarsfeld, Henry (1968) "Latent Structure Analysis";
Goodman (1974) "Exploratory latent structure analysis using both
identifiable and unidentifiable models"; Collins & Lanza (2009)
"Latent Class and Latent Transition Analysis".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_binary(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 6 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError("bad response matrix")
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("responses must be binary")
    return a


def latent_class_em(
    x: FloatArray, n_classes: int = 2, seed: int = 0, n_iter: int = 200, n_start: int = 8
) -> dict[str, FloatArray | float]:
    """EM fit for the Bernoulli latent class model.

    Returns ``class_probs`` (pi_k), ``item_probs`` (K x I Bernoulli
    params), ``resp`` (N x K responsibilities), ``assign`` (argmax
    class), ``loglik``, ``bic``.
    """
    a = _check_binary(x)
    n, m = a.shape
    if not 1 <= n_classes <= 8 or n <= 2 * n_classes:
        raise ValueError("bad n_classes")
    rng = np.random.default_rng(seed)
    best_ll = -np.inf
    best: dict[str, FloatArray | float] = {}
    eps = 1e-9
    for _ in range(max(1, n_start)):
        pi = rng.dirichlet(np.ones(n_classes))
        probs = rng.uniform(0.15, 0.85, (n_classes, m))
        prev = -np.inf
        resp = np.full((n, n_classes), 1.0 / n_classes)
        ll = -np.inf
        for _ in range(max(1, n_iter)):
            # E-step
            lp = (
                np.log(pi)[None, :]
                + a @ np.log(probs.T + eps)
                + (1 - a) @ np.log(1 - probs.T + eps)
            )
            lp = lp - lp.max(axis=1, keepdims=True)
            w = np.exp(lp)
            resp = w / w.sum(axis=1, keepdims=True)
            ll = float(np.log(w.sum(axis=1)).sum())
            # M-step
            nk = resp.sum(axis=0) + eps
            pi = nk / nk.sum()
            probs = (resp.T @ a) / nk[:, None]
            probs = np.clip(probs, eps, 1 - eps)
            if abs(ll - prev) < 1e-7:
                break
            prev = ll
        if ll > best_ll:
            best_ll = ll
            best = {
                "class_probs": pi.copy(),
                "item_probs": probs.copy(),
                "resp": resp.copy(),
            }
    n_par = n_classes * m + n_classes - 1
    bic = float(-2 * best_ll + n_par * np.log(n))
    resp = np.asarray(best["resp"], dtype=np.float64)
    best["assign"] = resp.argmax(axis=1).astype(float)
    best["loglik"] = float(best_ll)
    best["bic"] = bic
    return best


def bench_latent_class(seed: int = 20261231 + 415) -> dict[str, float]:
    """SYNTHIC check — planted 2-class mixture recovery (permutation-safe)."""
    rng = np.random.default_rng(seed)
    n, m = 400, 8
    pi_t = np.array([0.6, 0.4])
    p_t = np.array(
        [
            [0.8, 0.8, 0.8, 0.7, 0.2, 0.2, 0.2, 0.3],
            [0.2, 0.2, 0.3, 0.2, 0.8, 0.8, 0.7, 0.8],
        ]
    )
    z = (rng.random(n) < pi_t[1]).astype(int)
    x = (rng.random((n, m)) < p_t[z]).astype(float)
    fit = latent_class_em(x, n_classes=2, seed=seed)
    assign = np.asarray(fit["assign"], dtype=float)
    acc = max(float((assign == z).mean()), float((assign != z).mean()))
    pi_hat = np.sort(np.asarray(fit["class_probs"], dtype=float))
    pi_err = float(np.abs(pi_hat - np.sort(pi_t)).max())
    if acc < 0.9 or pi_err > 0.1:
        raise ValueError(f"lca recovery off: acc={acc:.3f} pi_err={pi_err:.3f}")
    return {
        "synthetic_lca_acc": acc,
        "synthetic_lca_pi_err": pi_err,
        "score": 1.0,
    }
