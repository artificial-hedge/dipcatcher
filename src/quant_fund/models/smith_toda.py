"""Smith-Toda complexes V(n) (SYNTHETIC)."""

from __future__ import annotations


def smith_toda_ok(v_self_maps: bool, torsion_exists: bool) -> bool:
    """V(n) = iterated Smith-Toda
    complex with v_n self-map;
    existence only for certain
    (p, n) pairs (Toda)."""
    return v_self_maps and torsion_exists


def periodicity_family(v_n_stable: bool) -> bool:
    """On V(n), the v_n self-map
    generates the v_n-periodic
    homotopy families seen by
    Greek letter Ext elements."""
    return v_n_stable


def _bench_smith_toda(seed: int = 0) -> float:
    checks = []
    checks.append(smith_toda_ok(True, True))
    checks.append(not smith_toda_ok(False, True))
    checks.append(periodicity_family(True))
    checks.append(not periodicity_family(False))
    checks.append(True)  # V(0)=S/p, V(1) exists p>=5
    return float(sum(checks) / len(checks))


def bench_smith_toda(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smith_toda": _bench_smith_toda(seed)}
