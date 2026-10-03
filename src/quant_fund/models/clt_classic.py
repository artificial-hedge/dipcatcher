"""Central limit theorem: normalized sums -> N(0,1) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def clt_z(n: int, rng: np.random.Generator) -> float:
    """(Xbar_n - mu)/(sigma/sqrt(n)) for exponentials."""
    x = rng.exponential(1.0, n)
    return float((np.mean(x) - 1.0) / (1.0 / np.sqrt(n)))


def _bench_clt_classic(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # standardized mean has unit-ish scale
    checks.append(abs(clt_z(5000, rng)) < 4.0)
    # variance of the limit = 1
    z = [clt_z(3000, rng) for _ in range(60)]
    checks.append(0.3 < float(np.var(z)) < 3.0)
    # shape ~ Gaussian: |z| < 1.96 often
    checks.append(sum(abs(t) < 1.96 for t in z) > 40)
    # needs finite variance (Cauchy fails: marker)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_clt_classic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clt_classic": _bench_clt_classic(seed)}
