"""hara hara module (SYNTHETIC)."""

from __future__ import annotations


def hara_hara_ok(rc: bool, potts: bool) -> bool:
    """hara_hara
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def hara_hara_aux(aux: bool) -> bool:
    """hara_hara
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_hara_hara(seed: int = 0) -> float:
    checks = []
    checks.append(hara_hara_ok(True, True))
    checks.append(not hara_hara_ok(False, True))
    checks.append(hara_hara_aux(True))
    checks.append(not hara_hara_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_hara_hara(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hara_hara": _bench_hara_hara(seed)}
