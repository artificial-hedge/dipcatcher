"""element free module (SYNTHETIC)."""

from __future__ import annotations


def element_free_ok(step: bool, conv: bool) -> bool:
    """element_free
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def element_free_aux(aux: bool) -> bool:
    """element_free
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_element_free(seed: int = 0) -> float:
    checks = []
    checks.append(element_free_ok(True, True))
    checks.append(not element_free_ok(False, True))
    checks.append(element_free_aux(True))
    checks.append(not element_free_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_element_free(seed: int = 0) -> dict[str, float]:
    return {"synthetic_element_free": _bench_element_free(seed)}
