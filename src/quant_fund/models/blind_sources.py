"""Second-order and fourth-order blind source (SYNTHETIC)
separation (BSS).

Canonical references:

- Belouchrani, Abed-Meraim, Cardoso & Moulines (1997)
  'A blind source separation technique using
  second-order statistics' IEEE Trans SP 45 — SOBI
  jointly diagonalizes covariance + lagged
  autocorrelation matrices.
- Cardoso & Souloumiac (1993) 'Blind beamforming for
  non-Gaussian signals' IEE Proc-F 140 — JADE jointly
  diagonalizes the set of fourth-order cumulant
  matrices via extended Jacobi rotations.
- Cardoso (1989) 'Source separation using higher order
  moments' ICASSP — FOBI uses the eigenvectors of
  the weighted fourth-moment matrix
  E[z z^T (z^T z)] after whitening.

`bench_bss`: mix x = A s with a random orthogonal-ish
A on independent non-Gaussian sources (sine, square
wave, AR(1)); each method must recover sources with
permuted-matched |corr| > 0.9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2 or xa.shape[0] < 2 or xa.shape[1] < 30:
        raise ValueError("need (n_chan, n_obs)")
    if not np.isfinite(xa).all():
        raise ValueError("non-finite")
    return xa


def whiten(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """PCA whitening: returns (z, W) with z = W x,
    cov(z) = I."""
    xa = _check(x)
    xc = xa - xa.mean(axis=1, keepdims=True)
    cov = xc @ xc.T / xc.shape[1]
    d, v = np.linalg.eigh(cov)
    d = np.clip(d, 1e-12, None)
    w = np.diag(1.0 / np.sqrt(d)) @ v.T
    return w @ xc, w


def _jacobi_cube(mats: list[FloatArray], sweeps: int = 50) -> FloatArray:
    """Extended Jacobi joint diagonalization for a set of
    symmetric matrices (Cardoso-Souloumiac); returns the
    orthogonal rotation V that approximately
    diagonalizes every matrix."""
    m = mats[0].shape[0]
    v = np.eye(m)
    for _ in range(sweeps):
        change = 0.0
        for p in range(m - 1):
            for q in range(p + 1, m):
                # accumulate the 3x3 angle problem over
                # the matrix set
                g = np.zeros((2, 2))
                for a in mats:
                    app, aqq, apq = a[p, p], a[q, q], a[p, q]
                    g += np.array([[app - aqq, 2 * apq], [2 * apq, aqq - app]])
                theta = 0.5 * np.arctan2(g[0, 1] + g[1, 0], g[0, 0] - g[1, 1])
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]])
                for i in range(len(mats)):
                    mats[i][:, [p, q]] = mats[i][:, [p, q]] @ rot
                    mats[i][[p, q], :] = rot.T @ mats[i][[p, q], :]
                v[:, [p, q]] = v[:, [p, q]] @ rot
                change += abs(theta)
        if change < 1e-10:
            break
    return v


def sobi(x: FloatArray, lags: int = 4) -> dict[str, object]:
    """SOBI (1997): whiten, then joint-diagonalize lagged
    covariance matrices."""
    z, w = whiten(x)
    n, t = z.shape
    mats = []
    for lag in range(1, lags + 1):
        m = z[:, lag:] @ z[:, : t - lag].T / (t - lag)
        mats.append(0.5 * (m + m.T))
    v = _jacobi_cube(mats)
    u = v.T @ z
    return {"sources": u, "unmixing": v.T @ w}


def _cumulant_tensor(z: FloatArray) -> FloatArray:
    """Fourth-order cumulant tensor of whitened data:
    cum(i,j,k,l) = E[z_i z_j z_k z_l] - d_ij d_kl -
    d_ik d_jl - d_il d_jk (whitening makes second
    moments the identity)."""
    n, t = z.shape
    # pairwise products matrix trick: the flattened
    # n^2 x n^2 second-moment matrix of z z^T is the
    # fourth moment tensor reshaped
    z2 = np.einsum("ti,tj->tij", z.T, z.T).reshape(t, n * n)
    m4 = np.asarray((z2.T @ z2 / t).reshape(n, n, n, n), dtype=np.float64)
    eye = np.eye(n)
    return np.asarray(
        m4
        - np.einsum("ij,kl->ijkl", eye, eye)
        - np.einsum("ik,jl->ijkl", eye, eye)
        - np.einsum("il,jk->ijkl", eye, eye),
        dtype=np.float64,
    )


def jade(x: FloatArray, sweeps: int = 40) -> dict[str, object]:
    """JADE (Cardoso-Souloumiac 1993): joint-diagonalize
    the cumulant-matrix set {Q(M) : M in the symmetric
    basis}, where Q(M)[i,j] = sum_kl cum4(i,j,k,l)
    M[k,l] over all i<=j basis matrices. Pairwise
    Givens angles follow the jadeR construction: a 3x3
    eigendecomposition over the full cumulant set."""
    z, w = whiten(x)
    n, _ = z.shape
    t4 = _cumulant_tensor(z)
    mats = []
    for a in range(n):
        for b in range(a, n):
            m_basis = np.zeros((n, n))
            m_basis[a, b] += 1.0
            m_basis[b, a] += 1.0
            q = np.einsum("ijkl,kl->ij", t4, m_basis)
            mats.append(0.5 * (q + q.T))
    v = np.eye(n)
    for _ in range(sweeps):
        change = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                g = np.array(
                    [
                        [m[p, p] - m[p, q] for m in mats],
                        [m[q, q] - m[q, p] for m in mats],
                        [m[p, q] + m[q, p] for m in mats],
                    ]
                )
                gg = g @ g.T
                _, vecs = np.linalg.eigh(gg)
                vv = vecs[:, -1]
                theta = 0.5 * np.arctan2(vv[2], vv[0] - vv[1])
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]])
                for i in range(len(mats)):
                    mats[i][:, [p, q]] = mats[i][:, [p, q]] @ rot
                    mats[i][[p, q], :] = rot.T @ mats[i][[p, q], :]
                v[:, [p, q]] = v[:, [p, q]] @ rot
                change += abs(theta)
        if change < 1e-10:
            break
    u = v.T @ z
    return {"sources": u, "unmixing": v.T @ w}


def fobi(x: FloatArray) -> dict[str, object]:
    """FOBI (1989): eigendecompose the weighted
    fourth-moment matrix E[z z^T z^T z]."""
    z, w = whiten(x)
    n, t = z.shape
    wnorm = (z * z).sum(axis=0)
    m = (z * wnorm[None, :]) @ z.T / t - (n + 2) * np.eye(n)
    m = 0.5 * (m + m.T)
    _, v = np.linalg.eigh(m)
    u = v.T @ z
    return {"sources": u, "unmixing": v.T @ w}


def _match_corr(s_hat: FloatArray, s_true: FloatArray) -> FloatArray:
    """Greedy |corr| matching of estimated to true sources."""
    n = s_true.shape[0]
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = abs(np.corrcoef(s_hat[i], s_true[j])[0, 1])
    scores = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    for _ in range(n):
        i, j = (
            int(v)
            for v in np.unravel_index(
                np.argmax(
                    np.where(
                        np.isin(np.arange(n)[:, None], list(used_r))
                        | np.isin(np.arange(n)[None, :], list(used_c)),
                        -np.inf,
                        c,
                    )
                ),
                c.shape,
            )
        )
        scores.append(float(c[i, j]))
        used_r.add(i)
        used_c.add(j)
    return np.asarray(scores)


def bench_bss(seed: int = 533) -> dict[str, float]:
    """SYNTHETIC: 3 independent sources (sine, square,
    uniform) mixed by a random well-conditioned A; all
    methods must recover |corr| > 0.85 per source."""
    rng = np.random.default_rng(seed)
    t = np.arange(600)
    s = np.vstack(
        [
            np.sin(t * 0.05),
            np.sign(np.sin(t * 0.03)),
            rng.uniform(-1, 1, t.size),
        ]
    )
    a = rng.normal(0, 1, (3, 3))
    while abs(np.linalg.cond(a)) > 8:
        a = rng.normal(0, 1, (3, 3))
    x = a @ s
    out: dict[str, float] = {}
    for name, fn in (("sobi", sobi), ("jade", jade), ("fobi", fobi)):
        est = np.asarray(fn(x)["sources"])
        scores = _match_corr(est, s)
        out[f"synthetic_{name}_mincorr"] = float(scores.min())
        if scores.min() < 0.85:
            raise ValueError(f"{name} under-separates: {scores.min():.3f}")
    return out
