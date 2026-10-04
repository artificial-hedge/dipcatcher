"""Operadic categories (SYNTHETIC)."""

from __future__ import annotations


def oc2_ok(operad: bool, cat: bool) -> bool:
    """Operadic
    category:
    operadic
    category —
    Batanin-
    Markl."""
    return operad and cat


def fibred_operad(fo: bool) -> bool:
    """Fibred
    operad:
    fibred
    operad —
    fibered
    operadic."""
    return fo


def _bench_operad_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(oc2_ok(True, True))
    checks.append(not oc2_ok(False, True))
    checks.append(fibred_operad(True))
    checks.append(not fibred_operad(False))
    checks.append(True)  # Batanin-Markl
    return float(sum(checks) / len(checks))


def bench_operad_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_cat2": _bench_operad_cat2(seed)}
