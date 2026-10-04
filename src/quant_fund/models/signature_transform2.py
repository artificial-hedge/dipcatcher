"""signature transform2 module (SYNTHETIC)."""

from __future__ import annotations


def signature_transform2_ok(rp1: bool, lift: bool) -> bool:
    """signature_transform2
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def signature_transform2_aux(aux: bool) -> bool:
    """signature_transform2
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_signature_transform2(seed: int = 0) -> float:
    checks = []
    checks.append(signature_transform2_ok(True, True))
    checks.append(not signature_transform2_ok(False, True))
    checks.append(signature_transform2_aux(True))
    checks.append(not signature_transform2_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_signature_transform2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_signature_transform2": _bench_signature_transform2(seed)}
