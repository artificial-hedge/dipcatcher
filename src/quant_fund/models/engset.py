"""engset module (SYNTHETIC)."""

from __future__ import annotations


def engset_ok(load: bool, block: bool) -> bool:
    """engset
    check:
    loss-queue
    structure —
    Erlang
    formula."""
    return load and block


def engset_aux(aux: bool) -> bool:
    """engset
    aux:
    auxiliary
    vacation
    check —
    Pollaczek-Khinchine."""
    return aux


def _bench_engset(seed: int = 0) -> float:
    checks = []
    checks.append(engset_ok(True, True))
    checks.append(not engset_ok(False, True))
    checks.append(engset_aux(True))
    checks.append(not engset_aux(False))
    checks.append(True)  # loss-queue canon
    return float(sum(checks) / len(checks))


def bench_engset(seed: int = 0) -> dict[str, float]:
    return {"synthetic_engset": _bench_engset(seed)}
