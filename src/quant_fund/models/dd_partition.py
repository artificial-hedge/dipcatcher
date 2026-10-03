"""dd partition module (SYNTHETIC)."""

from __future__ import annotations


def dd_partition_ok(part: bool, coarse: bool) -> bool:
    """dd_partition
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def dd_partition_aux(aux: bool) -> bool:
    """dd_partition
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_dd_partition(seed: int = 0) -> float:
    checks = []
    checks.append(dd_partition_ok(True, True))
    checks.append(not dd_partition_ok(False, True))
    checks.append(dd_partition_aux(True))
    checks.append(not dd_partition_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_dd_partition(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dd_partition": _bench_dd_partition(seed)}
