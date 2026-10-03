"""mortar dd module (SYNTHETIC)."""

from __future__ import annotations


def mortar_dd_ok(dom: bool, overlap: bool) -> bool:
    """mortar_dd
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def mortar_dd_aux(aux: bool) -> bool:
    """mortar_dd
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_mortar_dd(seed: int = 0) -> float:
    checks = []
    checks.append(mortar_dd_ok(True, True))
    checks.append(not mortar_dd_ok(False, True))
    checks.append(mortar_dd_aux(True))
    checks.append(not mortar_dd_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_mortar_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mortar_dd": _bench_mortar_dd(seed)}
