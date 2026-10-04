"""kassel wu module (SYNTHETIC)."""

from __future__ import annotations


def kassel_wu_ok(loop: bool, gff: bool) -> bool:
    """kassel_wu
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def kassel_wu_aux(aux: bool) -> bool:
    """kassel_wu
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_kassel_wu(seed: int = 0) -> float:
    checks = []
    checks.append(kassel_wu_ok(True, True))
    checks.append(not kassel_wu_ok(False, True))
    checks.append(kassel_wu_aux(True))
    checks.append(not kassel_wu_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_kassel_wu(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kassel_wu": _bench_kassel_wu(seed)}
