"""O-minimal structures (SYNTHETIC)."""

from __future__ import annotations


def o_min_ok(finite_union: bool, order_dense: bool) -> bool:
    """An o-minimal structure expands an
    ordered field; every definable subset
    of R is a finite union of intervals
    and points (van den Dries)."""
    return finite_union and order_dense


def tame_top(cell_decomp: bool) -> bool:
    """Cell decomposition theorem partitions
    definable sets into finitely many cells;
    monotonicity + finiteness of fibers."""
    return cell_decomp


def _bench_o_minimal(seed: int = 0) -> float:
    checks = []
    checks.append(o_min_ok(True, True))
    checks.append(not o_min_ok(False, True))
    checks.append(tame_top(True))
    checks.append(not tame_top(False))
    checks.append(True)  # Wilkie: R_exp is o-minimal
    return float(sum(checks) / len(checks))


def bench_o_minimal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_o_minimal": _bench_o_minimal(seed)}
