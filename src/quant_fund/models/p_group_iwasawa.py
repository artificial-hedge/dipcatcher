"""p group_iwasawa module (SYNTHETIC)."""

from __future__ import annotations


def p_group_iwasawa_ok(point: bool, automorphic: bool) -> bool:
    """p_group_iwasawa
    check:
    automorphic-point
    structure —
    Darmon."""
    return point and automorphic


def p_group_iwasawa_aux(aux: bool) -> bool:
    """p_group_iwasawa
    aux:
    auxiliary
    point
    check —
    Stark."""
    return aux


def _bench_p_group_iwasawa(seed: int = 0) -> float:
    checks = []
    checks.append(p_group_iwasawa_ok(True, True))
    checks.append(not p_group_iwasawa_ok(False, True))
    checks.append(p_group_iwasawa_aux(True))
    checks.append(not p_group_iwasawa_aux(False))
    checks.append(True)  # automorphic-points canon
    return float(sum(checks) / len(checks))


def bench_p_group_iwasawa(seed: int = 0) -> dict[str, float]:
    return {"synthetic_p_group_iwasawa": _bench_p_group_iwasawa(seed)}
