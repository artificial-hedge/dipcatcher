"""Degree-corrected stochastic block model (DC-SBM)
community inference.

- DC-SBM likelihood (Karrer & Newman 2011): Bernoulli
  model with per-block-pair rates e_rs and Poisson degree
  correction.
- Inference via regularized spectral clustering on the
  normalized Laplacian (Amini et al. 2013) followed by
  K-means on the leading eigenvectors.
- Model selection by held-in-likelihood / BIC over K.
- Community detection quality via normalized mutual
  information (Danon et al. 2005).

References
----------
- Karrer & Newman (2011) 'Stochastic blockmodels and
  community structure in networks' Phys. Rev. E 83.
- Amini et al. (2013) 'Pseudo-likelihood methods for
  community detection in large sparse networks' Ann.
  Statist. 41(4).
- Danon et al. (2005) 'Comparing community structure
  identification' J. Stat. Mech. P09008.
- Rohe, Chatterjee & Yu (2011) 'Spectral clustering and
  the high-dimensional SBM' Ann. Statist. 39(4).

Honesty
-------
SYNTHETIC self-check: a planted-partition graph where
recovery NMI is asserted > 0.9 vs a random-label null.

Composition
-----------
Pure numpy/scipy. Input is a symmetric adjacency matrix
(or edge list); outputs are labels, block rates, NMI.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as spla

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_adj(A: FloatArray) -> FloatArray:
    Aa = np.asarray(A, dtype=np.float64)
    if Aa.ndim != 2 or Aa.shape[0] != Aa.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(Aa, Aa.T):
        raise ValueError("adjacency must be symmetric")
    if (Aa < 0).any():
        raise ValueError("adjacency must be non-negative")
    return Aa


def _kmeans(pts: FloatArray, k: int, seed: int, iters: int = 50) -> IntArray:
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    idx = rng.choice(n, k, replace=False)
    cent = pts[idx].copy()
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((pts[:, None, :] - cent[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            members = pts[lab == j]
            if members.size:
                cent[j] = members.mean(0)
    return lab


def spectral_sbm(A: FloatArray, k: int, seed: int = 0, tau: float | None = None) -> IntArray:
    """Regularized spectral clustering for DC-SBM.

    L_tau = D_tau^{-1/2} (A + tau 11') D_tau^{-1/2}; cluster
    rows of the top-k eigenvector matrix.
    """
    Aa = _check_adj(A)
    n = Aa.shape[0]
    if not (1 < k < n):
        raise ValueError("k must be in (1, n)")
    d = Aa.sum(1)
    if tau is None:
        tau = float(d.mean())
    d_tau = d + tau
    dm = 1.0 / np.sqrt(d_tau)
    As = sparse.csr_matrix(Aa)
    L = sparse.diags(dm) @ (As + tau * np.ones((n, n))) @ sparse.diags(dm)
    vals, vecs = spla.eigsh(L, k=k, which="LA")
    order = np.argsort(-vals)
    emb = vecs[:, order]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return _kmeans(emb / norms, k, seed)


def dcsbm_loglik(A: FloatArray, z: IntArray) -> float:
    """Karrer-Newman DC-SBM log-likelihood (Poisson block rates)."""
    Aa = _check_adj(A)
    z = np.asarray(z, dtype=np.int64)
    k = int(z.max()) + 1
    n = Aa.shape[0]
    deg = Aa.sum(1)
    kappa = np.zeros(k)
    for i in range(n):
        kappa[z[i]] += deg[i]
    if (kappa <= 0).any():
        return -np.inf
    Z = np.zeros((n, k))
    Z[np.arange(n), z] = 1.0
    e_rs = Z.T @ Aa @ Z  # within-block diagonal counted twice (Poisson convention)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = e_rs / np.outer(kappa, kappa)
        log_term = np.where(e_rs > 0, e_rs * np.log(np.maximum(rate, 1e-300)), 0.0)
    ll = float(0.5 * (log_term.sum() - e_rs.sum()))
    return ll


def choose_k(A: FloatArray, k_max: int = 6, seed: int = 0) -> tuple[IntArray, FloatArray]:
    """BIC over K via DC-SBM log-likelihood."""
    Aa = _check_adj(A)
    n = Aa.shape[0]
    lls = np.full(k_max, -np.inf)
    for k in range(2, k_max + 1):
        z = spectral_sbm(Aa, k, seed)
        ll = dcsbm_loglik(Aa, z)
        n_edges = float(Aa.sum() / 2)
        n_params = k * (k + 1) / 2 + n  # block rates + degree params
        lls[k - 1] = ll - 0.5 * n_params * np.log(max(n_edges, 2))
    best_k = int(np.argmax(lls[1:]) + 2)
    return spectral_sbm(Aa, best_k, seed), lls


def nmi(labels_a: IntArray, labels_b: IntArray) -> float:
    """Normalized mutual information (max-normalization)."""
    a = np.asarray(labels_a, dtype=np.int64).ravel()
    b = np.asarray(labels_b, dtype=np.int64).ravel()
    if a.size != b.size or a.size == 0:
        raise ValueError("labels mismatch")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ka, kb = ai.max() + 1, bi.max() + 1
    n = a.size
    tab = np.zeros((ka, kb))
    np.add.at(tab, (ai, bi), 1)
    pa = tab.sum(1) / n
    pb = tab.sum(0) / n
    pab = tab / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            pab > 0,
            pab * np.log(pab / np.outer(pa, pb)),
            0.0,
        )
    mi = float(terms.sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha <= 0 or hb <= 0:
        return 0.0
    return mi / max(ha, hb)


def bench_sbm(seed: int = 506) -> dict[str, float]:
    """SYNTHETIC: 4-block planted partition, n=240, NMI>0.9 gate."""
    rng = np.random.default_rng(seed)
    n, k = 240, 4
    z_true = np.repeat(np.arange(k), n // k)
    # block probs: strong assortative structure
    B = np.full((k, k), 0.015)
    np.fill_diagonal(B, 0.22)
    P = B[z_true][:, z_true]
    U = rng.random((n, n))
    A = ((U < P) & (np.triu(np.ones((n, n))) == 1)).astype(np.float64)
    A = A + A.T
    z_hat = spectral_sbm(A, k, seed)
    score = nmi(z_true, z_hat)
    null = rng.permutation(z_true)
    null_nmi = nmi(z_true, null)
    if score < 0.9:
        raise ValueError("planted partition not recovered")
    z_bic, lls = choose_k(A, k_max=6, seed=seed)
    bic_k = int(np.argmax(lls[1:]) + 2)
    return {
        "synthetic_nmi": score,
        "synthetic_null_nmi": null_nmi,
        "synthetic_bic_k": float(bic_k),
        "synthetic_loglik": float(dcsbm_loglik(A, z_hat)),
        "synthetic_mean_deg": float(A.sum() / n),
    }
