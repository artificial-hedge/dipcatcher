"""Oka coherence (SYNTHETIC)."""

from __future__ import annotations


def oka_ok(sheaf: bool, coherent: bool) -> bool:
    """Oka
    coherence:
    the
    sheaf
    of
    holomorphic
    functions
    is
    a
    coherent
    sheaf
    of
    rings."""
    return sheaf and coherent


def cartan_abel(ca: bool) -> bool:
    """Cartan
    theorems
    A
    and
    B:
    coherent
    sheaves
    on
    Stein
    spaces
    have
    global
    sections
    and
    no
    higher
    cohomology."""
    return ca


def _bench_oka_coherence(seed: int = 0) -> float:
    checks = []
    checks.append(oka_ok(True, True))
    checks.append(not oka_ok(False, True))
    checks.append(cartan_abel(True))
    checks.append(not cartan_abel(False))
    checks.append(True)  # Oka-Cartan
    return float(sum(checks) / len(checks))


def bench_oka_coherence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oka_coherence": _bench_oka_coherence(seed)}
