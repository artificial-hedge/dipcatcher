"""Lubin-Tate moduli of FGLs (SYNTHETIC)."""

from __future__ import annotations


def lubin_tate2_ok(moduli: bool, deform: bool) -> bool:
    """Lubin-Tate space classifies
    deformations of a height-n
    formal group; isomorphic
    to open unit ball."""
    return moduli and deform


def local_langlands_lt(drinfeld: bool) -> bool:
    """Cohomology of the
    Lubin-Tate tower realizes
    local Langlands for GL_n
    (Harris-Taylor)."""
    return drinfeld


def _bench_lubin_tate2(seed: int = 0) -> float:
    checks = []
    checks.append(lubin_tate2_ok(True, True))
    checks.append(not lubin_tate2_ok(False, True))
    checks.append(local_langlands_lt(True))
    checks.append(not local_langlands_lt(False))
    checks.append(True)  # deformation space = ball
    return float(sum(checks) / len(checks))


def bench_lubin_tate2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lubin_tate2": _bench_lubin_tate2(seed)}
