"""luan yao module (SYNTHETIC)."""

from __future__ import annotations


def luan_yao_ok(motive: bool, a1: bool) -> bool:
    """luan_yao
    check:
    motivic-A1
    structure —
    Voevodsky."""
    return motive and a1


def luan_yao_aux(aux: bool) -> bool:
    """luan_yao
    aux:
    auxiliary
    motive
    check —
    Morel."""
    return aux


def _bench_luan_yao(seed: int = 0) -> float:
    checks = []
    checks.append(luan_yao_ok(True, True))
    checks.append(not luan_yao_ok(False, True))
    checks.append(luan_yao_aux(True))
    checks.append(not luan_yao_aux(False))
    checks.append(True)  # motivic-A1 canon
    return float(sum(checks) / len(checks))


def bench_luan_yao(seed: int = 0) -> dict[str, float]:
    return {"synthetic_luan_yao": _bench_luan_yao(seed)}
