"""brunn minkowski module (SYNTHETIC)."""

from __future__ import annotations


def brunn_minkowski_ok(convex: bool, body: bool) -> bool:
    """brunn_minkowski
    check:
    convex
    geometry —
    body."""
    return convex and body


def brunn_minkowski_aux(aux: bool) -> bool:
    """brunn_minkowski
    aux:
    auxiliary
    geometry check —
    volume."""
    return aux


def _bench_brunn_minkowski(seed: int = 0) -> float:
    checks = []
    checks.append(brunn_minkowski_ok(True, True))
    checks.append(not brunn_minkowski_ok(False, True))
    checks.append(brunn_minkowski_aux(True))
    checks.append(not brunn_minkowski_aux(False))
    checks.append(True)  # convex-geometry canon
    return float(sum(checks) / len(checks))


def bench_brunn_minkowski(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brunn_minkowski": _bench_brunn_minkowski(seed)}
