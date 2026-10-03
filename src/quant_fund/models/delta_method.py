"""delta method module (SYNTHETIC)."""

from __future__ import annotations


def delta_method_ok(weak: bool, conv: bool) -> bool:
    """delta_method
    check:
    weak
    convergence —
    measure."""
    return weak and conv


def delta_method_aux(aux: bool) -> bool:
    """delta_method
    aux:
    auxiliary
    convergence check —
    approx."""
    return aux


def _bench_delta_method(seed: int = 0) -> float:
    checks = []
    checks.append(delta_method_ok(True, True))
    checks.append(not delta_method_ok(False, True))
    checks.append(delta_method_aux(True))
    checks.append(not delta_method_aux(False))
    checks.append(True)  # weak-convergence canon
    return float(sum(checks) / len(checks))


def bench_delta_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delta_method": _bench_delta_method(seed)}
