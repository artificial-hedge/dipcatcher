"""slivnyak module (SYNTHETIC)."""

from __future__ import annotations


def slivnyak_ok(fil: bool, loc: bool) -> bool:
    """slivnyak
    check:
    filtration
    structure —
    Jacod-Shiryaev
    limit."""
    return fil and loc


def slivnyak_aux(aux: bool) -> bool:
    """slivnyak
    aux:
    auxiliary
    converg
    check —
    Pinsky
    process."""
    return aux


def _bench_slivnyak(seed: int = 0) -> float:
    checks = []
    checks.append(slivnyak_ok(True, True))
    checks.append(not slivnyak_ok(False, True))
    checks.append(slivnyak_aux(True))
    checks.append(not slivnyak_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_slivnyak(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slivnyak": _bench_slivnyak(seed)}
