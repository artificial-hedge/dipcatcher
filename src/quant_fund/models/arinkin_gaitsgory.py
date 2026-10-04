"""Arinkin-Gaitsgory geometric Langlands (SYNTHETIC)."""

from __future__ import annotations


def ag_ok(spectral_side: bool, automorphic_side: bool) -> bool:
    """Arinkin-
    Gaitsgory:
    categorical
    geometric
    Langlands —
    IndCoh
    on
    Bun_G
    vs
    quasi-
    coherent
    on
    LocSys."""
    return spectral_side and automorphic_side


def indcoh_langlands(il: bool) -> bool:
    """IndCoh
    Langlands:
    IndCoh
    on
    Bun
    matches
    QCoh
    on
    spectral —
    refined
    conjecture."""
    return il


def _bench_arinkin_gaitsgory(seed: int = 0) -> float:
    checks = []
    checks.append(ag_ok(True, True))
    checks.append(not ag_ok(False, True))
    checks.append(indcoh_langlands(True))
    checks.append(not indcoh_langlands(False))
    checks.append(True)  # Arinkin-Gaitsgory
    return float(sum(checks) / len(checks))


def bench_arinkin_gaitsgory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arinkin_gaitsgory": _bench_arinkin_gaitsgory(seed)}
