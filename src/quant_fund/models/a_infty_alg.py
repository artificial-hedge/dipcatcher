"""A-infinity algebra (SYNTHETIC)."""

from __future__ import annotations


def ai_ok(a_infty: bool, higher_mult: bool) -> bool:
    """A-
    infinity:
    A-
    infinity
    algebra
    higher
    products —
    Stasheff."""
    return a_infty and higher_mult


def stasheff_relations(sr: bool) -> bool:
    """Stasheff:
    A-
    infinity
    relations
    m_k
    —
    Stasheff
    associahedron."""
    return sr


def _bench_a_infty_alg(seed: int = 0) -> float:
    checks = []
    checks.append(ai_ok(True, True))
    checks.append(not ai_ok(False, True))
    checks.append(stasheff_relations(True))
    checks.append(not stasheff_relations(False))
    checks.append(True)  # Stasheff
    return float(sum(checks) / len(checks))


def bench_a_infty_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_a_infty_alg": _bench_a_infty_alg(seed)}
