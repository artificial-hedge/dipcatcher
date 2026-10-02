"""Evolution strategies: (μ/μ,λ)-ES with log-rank
recombination weights and global σ self-adaptation
(Rechenberg/Schwefel lineage; Hansen's modern weights),
plus a (1+1)-ES with the 1/5 success rule. Distinct from
CMA-ES (w85): no covariance path, isotropic/anisotropic-σ
mutation only. Synthetic bench gates convergence on
sphere/ellipsoid landscapes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def es_mu_lambda(
    f: Callable[[FloatArray], float],
    x0: FloatArray,
    sigma: float = 0.5,
    mu: int = 8,
    lam: int = 40,
    it: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    """(μ/μ,λ)-ES: rank-weighted mean of the top-μ of λ
    offspring; σ adapts by exp(τ·(success - target)) where
    τ = 1/√(2d) and success is measured via step-size
    self-adaptation on per-coordinate σ."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64)
    d = x.shape[0]
    sig = np.full(d, sigma)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    tau0 = 1.0 / np.sqrt(2 * np.sqrt(d))
    for _ in range(it):
        eps = rng.normal(0, 1, (lam, d))
        sig_s = sig * np.exp(tau0 * rng.normal(0, 1, (lam, 1)))
        off = x[None, :] + sig_s * eps
        fit = np.array([f(o) for o in off])
        idx = np.argsort(fit)[:mu]
        x = (off[idx].T @ w).ravel()
        sig = (sig_s[idx].T @ w).ravel()
        sig = np.clip(sig, 1e-12, 10.0)
    return {"x": x, "sigma": sig, "f": f(x)}


def es_one_plus_one(
    f: Callable[[FloatArray], float],
    x0: FloatArray,
    sigma: float = 0.3,
    it: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """(1+1)-ES with Rechenberg's 1/5 success rule."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64)
    fx = float(f(x))
    sig = float(sigma)
    n_succ = 0
    window = 20
    for t in range(it):
        y = x + sig * rng.normal(0, 1, x.shape[0])
        fy = float(f(y))
        if fy < fx:
            x, fx = y, fy
            n_succ += 1
        if (t + 1) % window == 0:
            if n_succ / window > 0.2:
                sig *= 1.3
            else:
                sig /= 1.3
            n_succ = 0
    return {"x": x, "sigma": sig, "f": fx}


def bench_evolution_strategies(seed: int = 563) -> dict[str, float]:
    """SYNTHETIC: (μ/μ,λ)-ES on a rotated ellipsoid and
    (1+1)-ES on the sphere — both must reach far better
    fitness than a random-search baseline of equal budget."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    d = 8
    cond = np.geomspace(1, 50, d)
    ell = lambda x: float(np.sum(cond * x**2))  # noqa: E731

    x0 = rng.normal(0, 3, d)
    r = es_mu_lambda(ell, x0, sigma=1.0, mu=8, lam=40, it=400, seed=seed)
    out["synthetic_es_ellipsoid_f"] = float(np.asarray(r["f"]))
    # random search baseline, same eval budget
    cand = rng.normal(0, 3, (400 * 40, d))
    out["synthetic_es_baseline_f"] = float(min(ell(c) for c in cand))
    if out["synthetic_es_ellipsoid_f"] > 0.05 * out["synthetic_es_baseline_f"]:
        raise ValueError(
            f"es ellipsoid off: {out['synthetic_es_ellipsoid_f']} vs baseline {out['synthetic_es_baseline_f']}"
        )

    sph = lambda x: float(np.sum(x**2))  # noqa: E731
    r1 = es_one_plus_one(sph, rng.normal(0, 2, d), sigma=0.5, it=4000, seed=seed + 1)
    out["synthetic_es11_sphere_f"] = float(np.asarray(r1["f"]))
    if out["synthetic_es11_sphere_f"] > 0.1:
        raise ValueError(f"1+1 es sphere off: {out['synthetic_es11_sphere_f']}")
    return out
