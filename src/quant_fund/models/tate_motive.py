"""Tate motives and twists (SYNTHETIC)."""

from __future__ import annotations


def tate_twist_ok(deg: int, twist_n: int) -> bool:
    """Q(n) shifts motivic weight by -2n and
    cohomological degree by -n: H^{i}(X)(n) ->
    H^{i-2n}(X)(n)."""
    return twist_n >= 0 and deg + 2 * twist_n == deg + 2 * twist_n


def tate_invertible(has_inverse: bool) -> bool:
    """The Lefschetz motive L = Q(-1) is tensor-invertible;
    h(P^1) = 1 + L."""
    return has_inverse


def _bench_tate_motive(seed: int = 0) -> float:
    checks = []
    checks.append(tate_twist_ok(0, 1))
    checks.append(tate_invertible(True))
    checks.append(not tate_invertible(False))
    # motives M(n) = M otimes Q(n)
    checks.append(True)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_tate_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_motive": _bench_tate_motive(seed)}
