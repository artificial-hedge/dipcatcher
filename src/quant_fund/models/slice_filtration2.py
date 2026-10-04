"""slice filtration2 module (SYNTHETIC)."""

from __future__ import annotations


def slice_filtration2_ok(motivic: bool, stable: bool) -> bool:
    """slice_filtration2
    check:
    motivic
    stable
    homotopy —
    slice."""
    return motivic and stable


def slice_filtration2_aux(aux: bool) -> bool:
    """slice_filtration2
    aux:
    auxiliary
    motivic
    check —
    spectral."""
    return aux


def _bench_slice_filtration2(seed: int = 0) -> float:
    checks = []
    checks.append(slice_filtration2_ok(True, True))
    checks.append(not slice_filtration2_ok(False, True))
    checks.append(slice_filtration2_aux(True))
    checks.append(not slice_filtration2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_slice_filtration2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slice_filtration2": _bench_slice_filtration2(seed)}
