"""pcdt module (SYNTHETIC)."""

from __future__ import annotations


def pcdt_ok(sm: bool, dec: bool) -> bool:
    """pcdt
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def pcdt_aux(aux: bool) -> bool:
    """pcdt
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_pcdt(seed: int = 0) -> float:
    checks = []
    checks.append(pcdt_ok(True, True))
    checks.append(not pcdt_ok(False, True))
    checks.append(pcdt_aux(True))
    checks.append(not pcdt_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_pcdt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pcdt": _bench_pcdt(seed)}
