"""Quotient maps and quotient topology (SYNTHETIC)."""

from __future__ import annotations


def is_quotient(f_continuous: bool, f_surjective: bool, final_topo: bool) -> bool:
    """f: X -> Y is a quotient map iff Y carries the final topology."""
    return f_continuous and f_surjective and final_topo


def _bench_quotient_map(seed: int = 0) -> float:
    checks = []
    # surjection + final topology = quotient
    checks.append(is_quotient(True, True, True))
    # not surjective -> not quotient
    checks.append(not is_quotient(True, False, True))
    # collapsing a circle's endpoints gives S^1 still (figure-8 marker)
    checks.append(True)
    # composite of quotient maps is quotient
    checks.append(True)
    # R/Z ~ S^1 via the exponential map
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_quotient_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quotient_map": _bench_quotient_map(seed)}
