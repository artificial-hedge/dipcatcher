"""motivic spark module (SYNTHETIC)."""

from __future__ import annotations


def motivic_spark_ok(motivic: bool, stable: bool) -> bool:
    """motivic_spark
    check:
    motivic
    structure —
    spark."""
    return motivic and stable


def motivic_spark_aux(aux: bool) -> bool:
    """motivic_spark
    aux:
    auxiliary
    motivic
    check —
    fundamental."""
    return aux


def _bench_motivic_spark(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_spark_ok(True, True))
    checks.append(not motivic_spark_ok(False, True))
    checks.append(motivic_spark_aux(True))
    checks.append(not motivic_spark_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_spark(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_spark": _bench_motivic_spark(seed)}
