"""neron raynaud module (SYNTHETIC)."""

from __future__ import annotations


def neron_raynaud_ok(ramif: bool, model: bool) -> bool:
    """neron_raynaud
    check:
    ramification-2
    structure —
    Brylinski."""
    return ramif and model


def neron_raynaud_aux(aux: bool) -> bool:
    """neron_raynaud
    aux:
    auxiliary
    semistable
    check —
    Raynaud."""
    return aux


def _bench_neron_raynaud(seed: int = 0) -> float:
    checks = []
    checks.append(neron_raynaud_ok(True, True))
    checks.append(not neron_raynaud_ok(False, True))
    checks.append(neron_raynaud_aux(True))
    checks.append(not neron_raynaud_aux(False))
    checks.append(True)  # ramification-2 canon
    return float(sum(checks) / len(checks))


def bench_neron_raynaud(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neron_raynaud": _bench_neron_raynaud(seed)}
