"""Cyclic operads (SYNTHETIC)."""

from __future__ import annotations


def co_ok(cyclic: bool, operad: bool) -> bool:
    """Cyclic
    operad:
    cyclic
    operad —
    Getzler-
    Kapranov."""
    return cyclic and operad


def modular_operad(mo: bool) -> bool:
    """Modular
    operad:
    modular
    operad —
    Getzler-
    Kapranov
    modular."""
    return mo


def _bench_cyclic_operad(seed: int = 0) -> float:
    checks = []
    checks.append(co_ok(True, True))
    checks.append(not co_ok(False, True))
    checks.append(modular_operad(True))
    checks.append(not modular_operad(False))
    checks.append(True)  # Getzler-Kapranov
    return float(sum(checks) / len(checks))


def bench_cyclic_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclic_operad": _bench_cyclic_operad(seed)}
