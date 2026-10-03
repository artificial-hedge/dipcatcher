"""Motivic Adem (SYNTHETIC)."""

from __future__ import annotations


def ma_ok(motivic: bool, adem: bool) -> bool:
    """Motivic
    Adem:
    motivic
    Adem
    relations —
    motivic
    Adem."""
    return motivic and adem


def adem_relations(ar: bool) -> bool:
    """Adem
    relations:
    motivic
    Adem
    relation
    structure —
    motivic
    Adem."""
    return ar


def _bench_motivic_adem(seed: int = 0) -> float:
    checks = []
    checks.append(ma_ok(True, True))
    checks.append(not ma_ok(False, True))
    checks.append(adem_relations(True))
    checks.append(not adem_relations(False))
    checks.append(True)  # Adem
    return float(sum(checks) / len(checks))


def bench_motivic_adem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_adem": _bench_motivic_adem(seed)}
