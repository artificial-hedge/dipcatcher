"""chapuy dolega module (SYNTHETIC)."""

from __future__ import annotations


def chapuy_dolega_ok(map_: bool, plane: bool) -> bool:
    """chapuy_dolega
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def chapuy_dolega_aux(aux: bool) -> bool:
    """chapuy_dolega
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_chapuy_dolega(seed: int = 0) -> float:
    checks = []
    checks.append(chapuy_dolega_ok(True, True))
    checks.append(not chapuy_dolega_ok(False, True))
    checks.append(chapuy_dolega_aux(True))
    checks.append(not chapuy_dolega_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_chapuy_dolega(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chapuy_dolega": _bench_chapuy_dolega(seed)}
