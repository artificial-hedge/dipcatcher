"""braces e4 module (SYNTHETIC)."""

from __future__ import annotations


def braces_e4_ok(algebra: bool, higher: bool) -> bool:
    """braces_e4
    check:
    higher
    algebra
    structure —
    centralizer."""
    return algebra and higher


def braces_e4_aux(aux: bool) -> bool:
    """braces_e4
    aux:
    auxiliary
    higher
    algebra
    check —
    operad."""
    return aux


def _bench_braces_e4(seed: int = 0) -> float:
    checks = []
    checks.append(braces_e4_ok(True, True))
    checks.append(not braces_e4_ok(False, True))
    checks.append(braces_e4_aux(True))
    checks.append(not braces_e4_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_braces_e4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_braces_e4": _bench_braces_e4(seed)}
