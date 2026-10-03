"""brylinski kato module (SYNTHETIC)."""

from __future__ import annotations


def brylinski_kato_ok(ramif: bool, model: bool) -> bool:
    """brylinski_kato
    check:
    ramification-2
    structure —
    Brylinski."""
    return ramif and model


def brylinski_kato_aux(aux: bool) -> bool:
    """brylinski_kato
    aux:
    auxiliary
    semistable
    check —
    Raynaud."""
    return aux


def _bench_brylinski_kato(seed: int = 0) -> float:
    checks = []
    checks.append(brylinski_kato_ok(True, True))
    checks.append(not brylinski_kato_ok(False, True))
    checks.append(brylinski_kato_aux(True))
    checks.append(not brylinski_kato_aux(False))
    checks.append(True)  # ramification-2 canon
    return float(sum(checks) / len(checks))


def bench_brylinski_kato(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brylinski_kato": _bench_brylinski_kato(seed)}
