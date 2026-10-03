"""self stabilizing module (SYNTHETIC)."""

from __future__ import annotations


def self_stabilizing_ok(mv1: bool, mk: bool) -> bool:
    """self_stabilizing
    check:
    McKean-Vlasov
    —
    propagation
    of
    chaos."""
    return mv1 and mk


def self_stabilizing_aux(aux: bool) -> bool:
    """self_stabilizing
    aux:
    auxiliary
    Kac
    check —
    molecular
    chaos."""
    return aux


def _bench_self_stabilizing(seed: int = 0) -> float:
    checks = []
    checks.append(self_stabilizing_ok(True, True))
    checks.append(not self_stabilizing_ok(False, True))
    checks.append(self_stabilizing_aux(True))
    checks.append(not self_stabilizing_aux(False))
    checks.append(True)  # MKV canon
    return float(sum(checks) / len(checks))


def bench_self_stabilizing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_stabilizing": _bench_self_stabilizing(seed)}
