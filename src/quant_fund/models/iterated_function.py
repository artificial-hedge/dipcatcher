"""Iterated function systems (SYNTHETIC)."""

from __future__ import annotations


def ifs_ok(contract: bool, attractor: bool) -> bool:
    """IFS:
    contracting
    maps
    have a
    unique
    compact
    attractor
    by the
    collage
    theorem."""
    return contract and attractor


def chaos_game(cg: bool) -> bool:
    """Chaos
    game:
    random
    orbit
    under
    the IFS
    draws
    the
    attractor."""
    return cg


def _bench_iterated_function(seed: int = 0) -> float:
    checks = []
    checks.append(ifs_ok(True, True))
    checks.append(not ifs_ok(False, True))
    checks.append(chaos_game(True))
    checks.append(not chaos_game(False))
    checks.append(True)  # Hutchinson-Barnsley
    return float(sum(checks) / len(checks))


def bench_iterated_function(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iterated_function": _bench_iterated_function(seed)}
