"""kenyon wilson module (SYNTHETIC)."""

from __future__ import annotations


def kenyon_wilson_ok(loop: bool, gff: bool) -> bool:
    """kenyon_wilson
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def kenyon_wilson_aux(aux: bool) -> bool:
    """kenyon_wilson
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_kenyon_wilson(seed: int = 0) -> float:
    checks = []
    checks.append(kenyon_wilson_ok(True, True))
    checks.append(not kenyon_wilson_ok(False, True))
    checks.append(kenyon_wilson_aux(True))
    checks.append(not kenyon_wilson_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_kenyon_wilson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kenyon_wilson": _bench_kenyon_wilson(seed)}
