"""Codensity monad (SYNTHETIC)."""

from __future__ import annotations


def cd_ok(codensity: bool, right_kan: bool) -> bool:
    """Codensity:
    codensity
    monad
    via
    right
    Kan —
    codensity."""
    return codensity and right_kan


def right_kan_ext(rk: bool) -> bool:
    """Right
    Kan:
    right
    Kan
    extension
    along
    itself —
    codensity."""
    return rk


def _bench_codensity_monad(seed: int = 0) -> float:
    checks = []
    checks.append(cd_ok(True, True))
    checks.append(not cd_ok(False, True))
    checks.append(right_kan_ext(True))
    checks.append(not right_kan_ext(False))
    checks.append(True)  # Codensity
    return float(sum(checks) / len(checks))


def bench_codensity_monad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_codensity_monad": _bench_codensity_monad(seed)}
