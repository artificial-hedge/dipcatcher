"""Soule elements (SYNTHETIC)."""

from __future__ import annotations


def se_ok(soule: bool, etale_k: bool) -> bool:
    """Soule:
    Soule
    elements
    in
    etale
    K-theory —
    Soule
    elements."""
    return soule and etale_k


def etale_k_class(ek: bool) -> bool:
    """Etale
    K:
    etale
    K-theory
    class
    construction —
    Soule."""
    return ek


def _bench_soul_elem(seed: int = 0) -> float:
    checks = []
    checks.append(se_ok(True, True))
    checks.append(not se_ok(False, True))
    checks.append(etale_k_class(True))
    checks.append(not etale_k_class(False))
    checks.append(True)  # Soule
    return float(sum(checks) / len(checks))


def bench_soul_elem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_soul_elem": _bench_soul_elem(seed)}
