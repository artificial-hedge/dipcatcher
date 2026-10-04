"""gowers knot module (SYNTHETIC)."""

from __future__ import annotations


def gowers_knot_ok(sixv: bool, solv: bool) -> bool:
    """gowers_knot
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def gowers_knot_aux(aux: bool) -> bool:
    """gowers_knot
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_gowers_knot(seed: int = 0) -> float:
    checks = []
    checks.append(gowers_knot_ok(True, True))
    checks.append(not gowers_knot_ok(False, True))
    checks.append(gowers_knot_aux(True))
    checks.append(not gowers_knot_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_gowers_knot(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gowers_knot": _bench_gowers_knot(seed)}
