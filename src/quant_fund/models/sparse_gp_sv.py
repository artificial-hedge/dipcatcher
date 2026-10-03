"""Sparse variational GP (Titsias 2009 / SVGP) — M inducing points,
Titsias collapsed bound on the fixture's regression view (fit z* =
pseudo-target via GP on X→y). Held-out NLL vs full dense GP.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._vi_synth import vi_data


def _k(Xa: np.ndarray, Xb: np.ndarray, ls: float, sf: float) -> np.ndarray:
    d2 = ((Xa[:, None] - Xb[None]) ** 2).sum(-1)
    return np.asarray(sf * np.exp(-0.5 * d2 / ls**2))


def bench_sparse_gp_sv(seed: int = 727, M: int = 20) -> dict[str, float]:
    X, y, Xt, yt, _wt = vi_data(seed)
    rng = np.random.default_rng(seed)
    yc = y.astype(float) - 0.5
    ytc = yt.astype(float) - 0.5
    ls, sf, sn2 = 1.2, 1.0, 0.05
    idx = rng.choice(len(X), M, replace=False)
    Z = X[idx]
    Kmm = _k(Z, Z, ls, sf) + 1e-6 * np.eye(M)
    Knm = _k(X, Z, ls, sf)
    # Titsias collapsed lower bound (trace term included)
    A = Knm @ np.linalg.solve(Kmm, Knm.T)
    B = np.eye(len(X)) * sn2 + A
    Bi_y = np.linalg.solve(B, yc)
    elbo = (
        -0.5 * yc @ Bi_y
        - 0.5 * np.linalg.slogdet(B)[1]
        - (np.trace(np.diag(_k(X, X, ls, sf)) - A)) / (2 * sn2)
        - 0.5 * len(X) * np.log(2 * np.pi)
    )
    # predictive via inducing posterior
    Ks = _k(Z, Xt, ls, sf)
    mid = np.linalg.solve(Kmm + Knm.T @ Knm / sn2, Knm.T @ yc / sn2)
    mu_t = Ks.T @ mid
    V = np.linalg.solve(Kmm + Knm.T @ Knm / sn2, Kmm)
    var_t = (
        np.diag(_k(Xt, Xt, ls, sf))
        - np.diag(Ks.T @ np.linalg.solve(Kmm, Ks))
        + np.diag(Ks.T @ V @ np.linalg.solve(Kmm, Ks))
        + sn2
    )
    var_t = np.clip(var_t, 1e-6, None)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * var_t) + 0.5 * (ytc - mu_t) ** 2 / var_t))
    # baseline: linear ridge fit NLL
    w = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ yc)
    resid = ytc - Xt @ w
    nll_r = float(np.mean(0.5 * np.log(2 * np.pi * resid.var()) + 0.5 * resid**2 / resid.var()))
    return {
        "synthetic_svgp_test_nll": nll,
        "synthetic_svgp_ridge_nll": nll_r,
        "synthetic_svgp_nll_gain": nll_r - nll,
        "synthetic_svgp_elbo": float(elbo),
        "torch_available": 0.0,
    }
