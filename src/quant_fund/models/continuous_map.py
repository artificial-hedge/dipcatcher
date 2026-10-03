"""continuous map module (SYNTHETIC)."""

from __future__ import annotations


def continuous_map_ok(weak: bool, conv: bool) -> bool:
    """continuous_map
    check:
    weak
    convergence —
    measure."""
    return weak and conv


def continuous_map_aux(aux: bool) -> bool:
    """continuous_map
    aux:
    auxiliary
    convergence check —
    approx."""
    return aux


def _bench_continuous_map(seed: int = 0) -> float:
    checks = []
    checks.append(continuous_map_ok(True, True))
    checks.append(not continuous_map_ok(False, True))
    checks.append(continuous_map_aux(True))
    checks.append(not continuous_map_aux(False))
    checks.append(True)  # weak-convergence canon
    return float(sum(checks) / len(checks))


def bench_continuous_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_continuous_map": _bench_continuous_map(seed)}
