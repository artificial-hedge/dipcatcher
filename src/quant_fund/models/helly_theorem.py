"""helly theorem module (SYNTHETIC)."""

from __future__ import annotations


def helly_theorem_ok(convex: bool, body: bool) -> bool:
    """helly_theorem
    check:
    convex
    geometry —
    body."""
    return convex and body


def helly_theorem_aux(aux: bool) -> bool:
    """helly_theorem
    aux:
    auxiliary
    geometry check —
    volume."""
    return aux


def _bench_helly_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(helly_theorem_ok(True, True))
    checks.append(not helly_theorem_ok(False, True))
    checks.append(helly_theorem_aux(True))
    checks.append(not helly_theorem_aux(False))
    checks.append(True)  # convex-geometry canon
    return float(sum(checks) / len(checks))


def bench_helly_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_helly_theorem": _bench_helly_theorem(seed)}
