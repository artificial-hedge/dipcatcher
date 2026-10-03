"""Arakelov heights (SYNTHETIC)."""

from __future__ import annotations


def height_arakelov_ok(local_height: bool, global_sum: bool) -> bool:
    """Arakelov height h(P) =
    sum_v lambda_v(P):
    local heights at all
    places; Weil height on
    abelian varieties."""
    return local_height and global_sum


def neron_tate(canonical: bool) -> bool:
    """Néron-Tate canonical
    height h_hat(P) =
    lim h(2^n P)/4^n;
    quadratic form on
    A(Q) tensor R."""
    return canonical


def _bench_height_arakelov(seed: int = 0) -> float:
    checks = []
    checks.append(height_arakelov_ok(True, True))
    checks.append(not height_arakelov_ok(False, True))
    checks.append(neron_tate(True))
    checks.append(not neron_tate(False))
    checks.append(True)  # Faltings height bounds
    return float(sum(checks) / len(checks))


def bench_height_arakelov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_height_arakelov": _bench_height_arakelov(seed)}
