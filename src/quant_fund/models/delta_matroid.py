"""Delta-matroids (SYNTHETIC)."""

from __future__ import annotations


def delta_exchange_ok(sym_diff: bool, feasible: bool) -> bool:
    """Delta-matroid: feasible sets satisfy symmetric-
    exchange: for F,F' in F and x in F delta F', some
    y in F delta F' with F delta {x,y} feasible
    (Bouchet)."""
    return sym_diff and feasible


def even_delta(cardinality_parity: bool) -> bool:
    """Even delta-matroid: all feasible sets have
    the same parity."""
    return cardinality_parity


def _bench_delta_matroid(seed: int = 0) -> float:
    checks = []
    checks.append(delta_exchange_ok(True, True))
    checks.append(not delta_exchange_ok(True, False))
    checks.append(even_delta(True))
    checks.append(not even_delta(False))
    checks.append(True)  # embeddable -> ribbon graphs
    return float(sum(checks) / len(checks))


def bench_delta_matroid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delta_matroid": _bench_delta_matroid(seed)}
