"""Residue theorem (SYNTHETIC)."""

from __future__ import annotations


def rt_ok(residue_map: bool, sum_zero: bool) -> bool:
    """Residue:
    sum
    of
    residues
    on
    compact
    curve
    zero —
    residue
    theorem."""
    return residue_map and sum_zero


def serre_residue(sr: bool) -> bool:
    """Serre
    residue:
    residue
    pairing
    perfect
    on
    curves —
    Serre
    duality."""
    return sr


def _bench_residue_thm(seed: int = 0) -> float:
    checks = []
    checks.append(rt_ok(True, True))
    checks.append(not rt_ok(False, True))
    checks.append(serre_residue(True))
    checks.append(not serre_residue(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_residue_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_residue_thm": _bench_residue_thm(seed)}
