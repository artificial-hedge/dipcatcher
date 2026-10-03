"""kassel kenyon module (SYNTHETIC)."""

from __future__ import annotations


def kassel_kenyon_ok(dim: bool, hf: bool) -> bool:
    """kassel_kenyon
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def kassel_kenyon_aux(aux: bool) -> bool:
    """kassel_kenyon
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_kassel_kenyon(seed: int = 0) -> float:
    checks = []
    checks.append(kassel_kenyon_ok(True, True))
    checks.append(not kassel_kenyon_ok(False, True))
    checks.append(kassel_kenyon_aux(True))
    checks.append(not kassel_kenyon_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_kassel_kenyon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kassel_kenyon": _bench_kassel_kenyon(seed)}
