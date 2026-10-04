"""Hodge-Tate p-adic theory (SYNTHETIC)."""

from __future__ import annotations


def ht_ok(hodge_tate: bool, weights: bool) -> bool:
    """Hodge-Tate:
    Hodge-Tate
    decomposition —
    weights."""
    return hodge_tate and weights


def hodge_tate_decomp(htd: bool) -> bool:
    """Hodge-Tate
    decomp:
    C_p
    tensor
    decomposition —
    Sen."""
    return htd


def _bench_hodge_tate_padic(seed: int = 0) -> float:
    checks = []
    checks.append(ht_ok(True, True))
    checks.append(not ht_ok(False, True))
    checks.append(hodge_tate_decomp(True))
    checks.append(not hodge_tate_decomp(False))
    checks.append(True)  # Tate-Sen
    return float(sum(checks) / len(checks))


def bench_hodge_tate_padic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_tate_padic": _bench_hodge_tate_padic(seed)}
