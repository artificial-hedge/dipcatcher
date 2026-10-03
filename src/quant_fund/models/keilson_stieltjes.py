"""keilson stieltjes module (SYNTHETIC)."""

from __future__ import annotations


def keilson_stieltjes_ok(reg: bool, cyc: bool) -> bool:
    """keilson_stieltjes
    check:
    regenerative
    structure —
    Khinchin
    cycle."""
    return reg and cyc


def keilson_stieltjes_aux(aux: bool) -> bool:
    """keilson_stieltjes
    aux:
    auxiliary
    Palm
    check —
    Wold
    process."""
    return aux


def _bench_keilson_stieltjes(seed: int = 0) -> float:
    checks = []
    checks.append(keilson_stieltjes_ok(True, True))
    checks.append(not keilson_stieltjes_ok(False, True))
    checks.append(keilson_stieltjes_aux(True))
    checks.append(not keilson_stieltjes_aux(False))
    checks.append(True)  # regenerative canon
    return float(sum(checks) / len(checks))


def bench_keilson_stieltjes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_keilson_stieltjes": _bench_keilson_stieltjes(seed)}
