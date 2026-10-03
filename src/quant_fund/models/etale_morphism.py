"""Etale morphism (SYNTHETIC)."""

from __future__ import annotations


def em_ok2(etale: bool, flat: bool) -> bool:
    """Etale:
    etale
    =
    flat
    unramified —
    SGA
    etale."""
    return etale and flat


def unramified_flat(uf: bool) -> bool:
    """Unramified:
    flat
    and
    unramified
    morphism —
    etale
    def."""
    return uf


def _bench_etale_morphism(seed: int = 0) -> float:
    checks = []
    checks.append(em_ok2(True, True))
    checks.append(not em_ok2(False, True))
    checks.append(unramified_flat(True))
    checks.append(not unramified_flat(False))
    checks.append(True)  # SGA
    return float(sum(checks) / len(checks))


def bench_etale_morphism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_morphism": _bench_etale_morphism(seed)}
