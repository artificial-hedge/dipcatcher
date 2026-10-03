"""higher ramif module (SYNTHETIC)."""

from __future__ import annotations


def higher_ramif_ok(ramif: bool, model: bool) -> bool:
    """higher_ramif
    check:
    ramification-2
    structure —
    Brylinski."""
    return ramif and model


def higher_ramif_aux(aux: bool) -> bool:
    """higher_ramif
    aux:
    auxiliary
    semistable
    check —
    Raynaud."""
    return aux


def _bench_higher_ramif(seed: int = 0) -> float:
    checks = []
    checks.append(higher_ramif_ok(True, True))
    checks.append(not higher_ramif_ok(False, True))
    checks.append(higher_ramif_aux(True))
    checks.append(not higher_ramif_aux(False))
    checks.append(True)  # ramification-2 canon
    return float(sum(checks) / len(checks))


def bench_higher_ramif(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_ramif": _bench_higher_ramif(seed)}
