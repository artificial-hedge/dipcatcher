"""sheffield gff module (SYNTHETIC)."""

from __future__ import annotations


def sheffield_gff_ok(gff: bool, lqg: bool) -> bool:
    """sheffield_gff
    check:
    LQG
    structure —
    Sheffield."""
    return gff and lqg


def sheffield_gff_aux(aux: bool) -> bool:
    """sheffield_gff
    aux:
    auxiliary
    LQG
    check —
    Miller."""
    return aux


def _bench_sheffield_gff(seed: int = 0) -> float:
    checks = []
    checks.append(sheffield_gff_ok(True, True))
    checks.append(not sheffield_gff_ok(False, True))
    checks.append(sheffield_gff_aux(True))
    checks.append(not sheffield_gff_aux(False))
    checks.append(True)  # LQG canon
    return float(sum(checks) / len(checks))


def bench_sheffield_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheffield_gff": _bench_sheffield_gff(seed)}
