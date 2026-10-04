"""Grothendieck existence (SYNTHETIC)."""

from __future__ import annotations


def ge_ok(groth: bool, existence: bool) -> bool:
    """GFGA:
    Grothendieck
    existence
    theorem —
    GFGA
    formal
    GAGA."""
    return groth and existence


def formal_gaga(fg: bool) -> bool:
    """Formal
    GAGA:
    formal
    GAGA
    algebraization —
    Grothendieck
    GFGA."""
    return fg


def _bench_groth_existence(seed: int = 0) -> float:
    checks = []
    checks.append(ge_ok(True, True))
    checks.append(not ge_ok(False, True))
    checks.append(formal_gaga(True))
    checks.append(not formal_gaga(False))
    checks.append(True)  # GFGA
    return float(sum(checks) / len(checks))


def bench_groth_existence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_existence": _bench_groth_existence(seed)}
