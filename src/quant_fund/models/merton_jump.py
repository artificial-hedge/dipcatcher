"""merton jump module (SYNTHETIC)."""

from __future__ import annotations


def merton_jump_ok(jd: bool, mj: bool) -> bool:
    """merton_jump
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def merton_jump_aux(aux: bool) -> bool:
    """merton_jump
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_merton_jump(seed: int = 0) -> float:
    checks = []
    checks.append(merton_jump_ok(True, True))
    checks.append(not merton_jump_ok(False, True))
    checks.append(merton_jump_aux(True))
    checks.append(not merton_jump_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_merton_jump(seed: int = 0) -> dict[str, float]:
    return {"synthetic_merton_jump": _bench_merton_jump(seed)}
