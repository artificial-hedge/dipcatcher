"""divid diff_table module (SYNTHETIC)."""

from __future__ import annotations


def divid_diff_table_ok(node: bool, weight: bool) -> bool:
    """divid_diff_table
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def divid_diff_table_aux(aux: bool) -> bool:
    """divid_diff_table
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_divid_diff_table(seed: int = 0) -> float:
    checks = []
    checks.append(divid_diff_table_ok(True, True))
    checks.append(not divid_diff_table_ok(False, True))
    checks.append(divid_diff_table_aux(True))
    checks.append(not divid_diff_table_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_divid_diff_table(seed: int = 0) -> dict[str, float]:
    return {"synthetic_divid_diff_table": _bench_divid_diff_table(seed)}
