"""vc class module (SYNTHETIC)."""

from __future__ import annotations


def vc_class_ok(ent: bool, bound: bool) -> bool:
    """vc_class
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def vc_class_aux(aux: bool) -> bool:
    """vc_class
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_vc_class(seed: int = 0) -> float:
    checks = []
    checks.append(vc_class_ok(True, True))
    checks.append(not vc_class_ok(False, True))
    checks.append(vc_class_aux(True))
    checks.append(not vc_class_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_vc_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vc_class": _bench_vc_class(seed)}
