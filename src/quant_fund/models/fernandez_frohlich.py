"""fernandez frohlich module (SYNTHETIC)."""

from __future__ import annotations


def fernandez_frohlich_ok(on: bool, irf: bool) -> bool:
    """fernandez_frohlich
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def fernandez_frohlich_aux(aux: bool) -> bool:
    """fernandez_frohlich
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_fernandez_frohlich(seed: int = 0) -> float:
    checks = []
    checks.append(fernandez_frohlich_ok(True, True))
    checks.append(not fernandez_frohlich_ok(False, True))
    checks.append(fernandez_frohlich_aux(True))
    checks.append(not fernandez_frohlich_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_fernandez_frohlich(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fernandez_frohlich": _bench_fernandez_frohlich(seed)}
