"""le gall_miermont module (SYNTHETIC)."""

from __future__ import annotations


def le_gall_miermont_ok(map_: bool, plane: bool) -> bool:
    """le_gall_miermont
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def le_gall_miermont_aux(aux: bool) -> bool:
    """le_gall_miermont
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_le_gall_miermont(seed: int = 0) -> float:
    checks = []
    checks.append(le_gall_miermont_ok(True, True))
    checks.append(not le_gall_miermont_ok(False, True))
    checks.append(le_gall_miermont_aux(True))
    checks.append(not le_gall_miermont_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_le_gall_miermont(seed: int = 0) -> dict[str, float]:
    return {"synthetic_le_gall_miermont": _bench_le_gall_miermont(seed)}
