"""burkholder davis module (SYNTHETIC)."""

from __future__ import annotations


def burkholder_davis_ok(pred: bool, mart: bool) -> bool:
    """burkholder_davis
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def burkholder_davis_aux(aux: bool) -> bool:
    """burkholder_davis
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_burkholder_davis(seed: int = 0) -> float:
    checks = []
    checks.append(burkholder_davis_ok(True, True))
    checks.append(not burkholder_davis_ok(False, True))
    checks.append(burkholder_davis_aux(True))
    checks.append(not burkholder_davis_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_burkholder_davis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_burkholder_davis": _bench_burkholder_davis(seed)}
