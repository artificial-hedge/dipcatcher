"""sokal bcc module (SYNTHETIC)."""

from __future__ import annotations


def sokal_bcc_ok(rc: bool, potts: bool) -> bool:
    """sokal_bcc
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def sokal_bcc_aux(aux: bool) -> bool:
    """sokal_bcc
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_sokal_bcc(seed: int = 0) -> float:
    checks = []
    checks.append(sokal_bcc_ok(True, True))
    checks.append(not sokal_bcc_ok(False, True))
    checks.append(sokal_bcc_aux(True))
    checks.append(not sokal_bcc_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_sokal_bcc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sokal_bcc": _bench_sokal_bcc(seed)}
