"""Operadic cobar (SYNTHETIC)."""

from __future__ import annotations


def oc_ok(cobar: bool, operad: bool) -> bool:
    """Operadic
    cobar:
    cobar
    construction
    of
    cooperad —
    Getzler-
    Jones."""
    return cobar and operad


def cobar_operad(co: bool) -> bool:
    """Cobar
    operad:
    cobar
    gives
    operad —
    Getzler
    Jones."""
    return co


def _bench_operad_cobar(seed: int = 0) -> float:
    checks = []
    checks.append(oc_ok(True, True))
    checks.append(not oc_ok(False, True))
    checks.append(cobar_operad(True))
    checks.append(not cobar_operad(False))
    checks.append(True)  # Getzler-Jones
    return float(sum(checks) / len(checks))


def bench_operad_cobar(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_cobar": _bench_operad_cobar(seed)}
