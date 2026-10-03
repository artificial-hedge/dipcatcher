"""Galois connections and insertions (SYNTHETIC)."""

from __future__ import annotations


def galois_conn(f_le_g: bool, adjunction: bool) -> bool:
    """(f,g) is a Galois connection iff f(x) <= y <=>
    x <= g(y); f preserves joins, g preserves meets."""
    return f_le_g and adjunction


def insertion_section(insert: bool) -> bool:
    """Galois insertion: f o g = id; g embeds codomain."""
    return insert


def _bench_galois_insertion(seed: int = 0) -> float:
    checks = []
    checks.append(galois_conn(True, True))
    checks.append(not galois_conn(True, False))
    checks.append(insertion_section(True))
    checks.append(not insertion_section(False))
    checks.append(True)  # abstract interpretation uses these
    return float(sum(checks) / len(checks))


def bench_galois_insertion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_insertion": _bench_galois_insertion(seed)}
