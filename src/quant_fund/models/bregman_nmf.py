"""Bregman NMF: Itakura-Saito divergence vs Euclidean on Poisson counts.

IS-NMF uses the multiplicative updates for D_IS(V|WH) = sum v/wh - log(v/wh) - 1,
the right geometry for Poisson-generated data. Euclidean NMF is the planted
honest-negative baseline. Bench: held-likelihood (Poisson deviance) of the
IS factorization vs the Gaussian-noise one.
"""

import numpy as np

from quant_fund.models._ig_synth import poisson_nmf


def _is_nmf(
    V: np.ndarray, r: int, iters: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.random((V.shape[0], r)) + 0.1
    h = rng.random((r, V.shape[1])) + 0.1
    for _ in range(iters):
        lam = w @ h + 1e-9
        h *= np.sqrt((w.T @ (V / lam**2)) / (w.T @ (1.0 / lam)))
        lam = w @ h + 1e-9
        w *= np.sqrt((V / lam**2) @ h.T / ((1.0 / lam) @ h.T))
        scale = np.maximum(w.sum(axis=0), 1e-9)
        w /= scale
        h *= scale[:, None]
        w = np.clip(w, 1e-9, 1e6)
        h = np.clip(h, 1e-9, 1e6)
    return w, h


def _eu_nmf(
    V: np.ndarray, r: int, iters: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.random((V.shape[0], r)) + 0.1
    h = rng.random((r, V.shape[1])) + 0.1
    for _ in range(iters):
        h *= (w.T @ V) / (w.T @ w @ h + 1e-9)
        w *= (V @ h.T) / (w @ h @ h.T + 1e-9)
    return w, h


def _poisson_dev(V: np.ndarray, lam: np.ndarray) -> float:
    lam = lam + 1e-9
    return float(np.sum(V * np.log(V / lam) - V + lam))


def _is_div(V: np.ndarray, lam: np.ndarray) -> float:
    lam = lam + 1e-9
    return float(np.sum(V / lam - np.log(V / lam) - 1.0))


def bench_bregman_nmf(seed: int = 4307, r: int = 4) -> dict[str, float]:
    V = poisson_nmf(seed)
    w_is, h_is = _is_nmf(V, r)
    w_eu, h_eu = _eu_nmf(V, r)
    is_is = _is_div(V, w_is @ h_is)
    eu_is = _is_div(V, w_eu @ h_eu)
    d_is = _poisson_dev(V, w_is @ h_is)
    d_eu = _poisson_dev(V, w_eu @ h_eu)
    return {
        "synthetic_is_isdiv": is_is,
        "synthetic_eu_isdiv": eu_is,
        "synthetic_is_isdiv_gain": eu_is - is_is,
        "synthetic_is_deviance": d_is,
        "synthetic_eu_deviance": d_eu,
    }
