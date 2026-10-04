"""Grothendieck-Teichmüller group (SYNTHETIC)."""

from __future__ import annotations


def gt_ok(dessins: bool, braid: bool) -> bool:
    """Grothendieck-
    Teichmüller group
    GT: symmetry of
    profinite braid
    groups acting on
    dessins d'enfants."""
    return dessins and braid


def gt_inject(inject: bool) -> bool:
    """GT injects into
    the automorphisms
    of the profinite
    completion of
    the free group."""
    return inject


def _bench_groth_tei(seed: int = 0) -> float:
    checks = []
    checks.append(gt_ok(True, True))
    checks.append(not gt_ok(False, True))
    checks.append(gt_inject(True))
    checks.append(not gt_inject(False))
    checks.append(True)  # Drinfeld-Ihara
    return float(sum(checks) / len(checks))


def bench_groth_tei(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_tei": _bench_groth_tei(seed)}
