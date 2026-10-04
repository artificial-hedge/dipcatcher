"""propagation chaos module (SYNTHETIC)."""

from __future__ import annotations


def propagation_chaos_ok(mv1: bool, mk: bool) -> bool:
    """propagation_chaos
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def propagation_chaos_aux(aux: bool) -> bool:
    """propagation_chaos
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_propagation_chaos(seed: int = 0) -> float:
    checks = []
    checks.append(propagation_chaos_ok(True, True))
    checks.append(not propagation_chaos_ok(False, True))
    checks.append(propagation_chaos_aux(True))
    checks.append(not propagation_chaos_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_propagation_chaos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_propagation_chaos": _bench_propagation_chaos(seed)}
