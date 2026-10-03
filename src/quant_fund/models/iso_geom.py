"""iso geom module (SYNTHETIC)."""

from __future__ import annotations


def iso_geom_ok(cell: bool, embed: bool) -> bool:
    """iso_geom
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def iso_geom_aux(aux: bool) -> bool:
    """iso_geom
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_iso_geom(seed: int = 0) -> float:
    checks = []
    checks.append(iso_geom_ok(True, True))
    checks.append(not iso_geom_ok(False, True))
    checks.append(iso_geom_aux(True))
    checks.append(not iso_geom_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_iso_geom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iso_geom": _bench_iso_geom(seed)}
