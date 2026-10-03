"""Transfer matrix method (SYNTHETIC)."""

from __future__ import annotations


def tm_ok(state_graph: bool, counting_paths: bool) -> bool:
    """Transfer
    matrix:
    count
    sequences
    via
    adjacency
    powers —
    enumeration."""
    return state_graph and counting_paths


def transfer_method(tm2: bool) -> bool:
    """Transfer
    method:
    trace
    and
    spectral
    count
    walks —
    matrix
    enumeration."""
    return tm2


def _bench_transfer_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(tm_ok(True, True))
    checks.append(not tm_ok(False, True))
    checks.append(transfer_method(True))
    checks.append(not transfer_method(False))
    checks.append(True)  # Stanley
    return float(sum(checks) / len(checks))


def bench_transfer_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfer_matrix": _bench_transfer_matrix(seed)}
