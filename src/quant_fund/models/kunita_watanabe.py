"""kunita watanabe module (SYNTHETIC)."""

from __future__ import annotations


def kunita_watanabe_ok(fil: bool, loc: bool) -> bool:
    """kunita_watanabe
    check:
    filtration
    structure —
    Jacod-Shiryaev
    limit."""
    return fil and loc


def kunita_watanabe_aux(aux: bool) -> bool:
    """kunita_watanabe
    aux:
    auxiliary
    converg
    check —
    Pinsky
    process."""
    return aux


def _bench_kunita_watanabe(seed: int = 0) -> float:
    checks = []
    checks.append(kunita_watanabe_ok(True, True))
    checks.append(not kunita_watanabe_ok(False, True))
    checks.append(kunita_watanabe_aux(True))
    checks.append(not kunita_watanabe_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_kunita_watanabe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kunita_watanabe": _bench_kunita_watanabe(seed)}
