"""Weak law of large numbers (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def sample_mean_error(n: int, rng: np.random.Generator) -> float:
    x = rng.standard_normal(n)
    return float(abs(float(np.mean(x))))


def _bench_weak_law(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # sample mean concentrates at 0 as n grows
    checks.append(sample_mean_error(20000, rng) < 0.05)
    # error shrinks like 1/sqrt(n): E|xbar| ~ sigma*sqrt(2/(pi*n)) —
    # 50-trial mean at n=100 (~0.08) vs n=40000 (~0.004) differs ~20x.
    e_small = float(np.mean([sample_mean_error(100, rng) for _ in range(50)]))
    e_big = float(np.mean([sample_mean_error(40000, rng) for _ in range(50)]))
    checks.append(e_big < e_small / 5.0)
    # works for any finite mean: shift to 3.7 and the error still decays
    x_shift = 3.7 + rng.standard_normal(20000)
    checks.append(abs(float(np.mean(x_shift)) - 3.7) < 0.05)
    # convergence is in probability: exceedance of a fixed eps falls
    # with n (P ~ exp(-n eps^2/2) for Gaussians).
    eps_p = 0.05
    exceed_small = sum(sample_mean_error(100, rng) > eps_p for _ in range(100))
    exceed_big = sum(sample_mean_error(40000, rng) > eps_p for _ in range(100))
    checks.append(exceed_big < exceed_small)
    # iid hypothesis needed: AR(1) phi=0.9 has effective n ~ n/19, so
    # its mean error sits ~sqrt(19) above the iid error at equal n.
    ar_errs = []
    for _ in range(20):
        ar = np.zeros(400)
        for t in range(1, 400):
            ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
        ar_errs.append(abs(float(np.mean(ar))))
    iid_errs = [sample_mean_error(400, rng) for _ in range(20)]
    checks.append(float(np.mean(ar_errs)) > 3.0 * float(np.mean(iid_errs)))
    return float(sum(checks) / len(checks))


def bench_weak_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_law": _bench_weak_law(seed)}
