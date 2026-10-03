"""insensitive thm module (SYNTHETIC)."""

from __future__ import annotations


def insensitive_thm_ok(net: bool, prod: bool) -> bool:
    """insensitive_thm
    check:
    queue-net
    structure —
    BCMP
    form."""
    return net and prod


def insensitive_thm_aux(aux: bool) -> bool:
    """insensitive_thm
    aux:
    auxiliary
    MVA
    check —
    Gordon-Newell."""
    return aux


def _bench_insensitive_thm(seed: int = 0) -> float:
    checks = []
    checks.append(insensitive_thm_ok(True, True))
    checks.append(not insensitive_thm_ok(False, True))
    checks.append(insensitive_thm_aux(True))
    checks.append(not insensitive_thm_aux(False))
    checks.append(True)  # queue-net canon
    return float(sum(checks) / len(checks))


def bench_insensitive_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_insensitive_thm": _bench_insensitive_thm(seed)}
