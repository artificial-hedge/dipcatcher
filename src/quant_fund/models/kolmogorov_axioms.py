"""Kolmogorov axioms on a finite probability space (SYNTHETIC)."""

from __future__ import annotations

Prob = dict[int, float]


def prob_of(p: Prob, event: set[int]) -> float:
    return sum(p.get(w, 0.0) for w in event)


def is_probability(p: Prob) -> bool:
    return all(v >= 0 for v in p.values()) and abs(sum(p.values()) - 1.0) < 1e-9


def _bench_kolmogorov_axioms(seed: int = 0) -> float:
    checks = []
    p = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    checks.append(is_probability(p))
    omega = set(p)
    checks.append(abs(prob_of(p, omega) - 1.0) < 1e-9)
    checks.append(prob_of(p, set()) == 0.0)
    # finite additivity on disjoint events
    checks.append(abs(prob_of(p, {0, 1}) + prob_of(p, {2, 3}) - 1.0) < 1e-9)
    # complement rule
    checks.append(abs(prob_of(p, omega - {0, 1}) - (1 - prob_of(p, {0, 1}))) < 1e-9)
    # inclusion-exclusion
    a, b = {0, 1}, {1, 2}
    lhs = prob_of(p, a | b)
    rhs = prob_of(p, a) + prob_of(p, b) - prob_of(p, a & b)
    checks.append(abs(lhs - rhs) < 1e-9)
    # monotonicity
    checks.append(prob_of(p, {0}) <= prob_of(p, {0, 1}))
    # not a probability: sums to 1.1
    checks.append(not is_probability({0: 0.5, 1: 0.6}))
    return float(sum(checks) / len(checks))


def bench_kolmogorov_axioms(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_axioms": _bench_kolmogorov_axioms(seed)}
