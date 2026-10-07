"""Radon-Nikodym derivative estimation (wave 288) (SYNTHETIC).

d mu/d nu on [0,1] where mu = Beta(2,5), nu = uniform: density ratio via
binned-sample histogram ratio matches the beta pdf.
"""

import numpy as np

_SEED = 20261231 + 817


def _beta_pdf(x: np.ndarray) -> np.ndarray:
    import math

    c = math.gamma(7) / (math.gamma(2) * math.gamma(5))
    out: np.ndarray = c * x * (1 - x) ** 4
    return out


def rn_estimate(
    mu_s: np.ndarray, nu_s: np.ndarray, bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    hm, _ = np.histogram(mu_s, bins=edges, density=True)
    hn, _ = np.histogram(nu_s, bins=edges, density=True)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return mid, hm / np.clip(hn, 1e-9, None)


def bench_radon_nikodym(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    mu_s = rng.beta(2, 5, 120000)
    nu_s = rng.rand(120000)
    mid, ratio = rn_estimate(mu_s, nu_s)
    want = _beta_pdf(mid)  # nu is uniform density 1 on [0,1]
    mask = (mid > 0.05) & (mid < 0.9)
    rel = np.abs(ratio[mask] - want[mask]) / want[mask]
    return {"synthetic_rn_deriv": float(rel.mean() < 0.05)}
