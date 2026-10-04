"""Schlichting K-theory (SYNTHETIC)."""

from __future__ import annotations


def sk_ok(schlichting: bool, k: bool) -> bool:
    """Schlichting
    K:
    Schlichting
    K
    theory —
    dualities."""
    return schlichting and k


def dualities_k(dk: bool) -> bool:
    """Dualities
    K:
    dualities
    K
    theory —
    hermitian."""
    return dk


def _bench_schlichting_k(seed: int = 0) -> float:
    checks = []
    checks.append(sk_ok(True, True))
    checks.append(not sk_ok(False, True))
    checks.append(dualities_k(True))
    checks.append(not dualities_k(False))
    checks.append(True)  # Schlichting
    return float(sum(checks) / len(checks))


def bench_schlichting_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schlichting_k": _bench_schlichting_k(seed)}
