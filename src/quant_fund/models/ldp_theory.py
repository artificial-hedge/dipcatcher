"""Large-deviation rate functions: Cramer for Bernoulli (SYNTHETIC)."""

from __future__ import annotations

import math


def cramer_bernoulli(x: float, p: float) -> float:
    """Rate I(x) = x log(x/p) + (1-x) log((1-x)/(1-p)) for the sample
    mean of Bernoulli(p)."""
    if x in (0.0, 1.0):
        if x == 0.0:
            return math.log(1.0 / (1 - p))
        return math.log(1.0 / p)
    return x * math.log(x / p) + (1 - x) * math.log((1 - x) / (1 - p))


def contraction(rate_f, x_map: float) -> float:
    """Rate for the mapped event is the rate at the preimage level;
    here apply rate_f at the mapped point."""
    return float(rate_f(x_map))


def _bench_ldp_theory(seed: int = 0) -> float:
    checks = []
    # rate at the mean is zero
    checks.append(abs(cramer_bernoulli(0.5, 0.5)) < 1e-12)
    # rate positive off the mean
    checks.append(cramer_bernoulli(0.8, 0.5) > 0.1)
    # symmetric for p = 0.5
    checks.append(abs(cramer_bernoulli(0.8, 0.5) - cramer_bernoulli(0.2, 0.5)) < 1e-12)
    # asymmetric for p != 0.5: rate larger above mean of p=0.3 -> 0.9
    checks.append(cramer_bernoulli(0.9, 0.3) > cramer_bernoulli(0.9, 0.5))
    # endpoints: I(0) = -log(1-p), I(1) = -log(p)
    checks.append(abs(cramer_bernoulli(0.0, 0.5) - math.log(2)) < 1e-12)
    # contraction: mapped point rate equals preimage rate
    checks.append(
        contraction(lambda t: cramer_bernoulli(t, 0.5), 0.8) == cramer_bernoulli(0.8, 0.5)
    )
    return float(sum(checks) / len(checks))


def bench_ldp_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ldp_theory": _bench_ldp_theory(seed)}
