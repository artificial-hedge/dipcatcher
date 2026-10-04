"""Hilbert function/series of monomial ideals via staircase counting (SYNTHETIC)."""

from __future__ import annotations


def divides(m: tuple[int, ...], g: tuple[int, ...]) -> bool:
    return all(x <= y for x, y in zip(m, g, strict=True))


def standard_monomials(gens: list[tuple[int, ...]], deg: int, n_vars: int) -> list[tuple[int, ...]]:
    """Monomials of total degree `deg` not divisible by any generator."""
    out = []

    def rec(i: int, rem: int, cur: list[int]):
        if i == n_vars - 1:
            cur.append(rem)
            m = tuple(cur)
            if not any(divides(g, m) for g in gens):
                out.append(m)
            cur.pop()
            return
        for e in range(rem + 1):
            cur.append(e)
            rec(i + 1, rem - e, cur)
            cur.pop()

    rec(0, deg, [])
    return out


def hilbert_fn(gens: list[tuple[int, ...]], deg: int, n_vars: int) -> int:
    return len(standard_monomials(gens, deg, n_vars))


def _bench_hilbert_series(seed: int = 0) -> float:
    checks = []
    # k[x,y]: Hilbert function = deg+1
    checks.append(hilbert_fn([], 3, 2) == 4)
    # ideal (x^2) in k[x,y]: standard monomials of deg d: {x^a y^b: a<2} -> 2 if d>=1
    checks.append(hilbert_fn([(2, 0)], 3, 2) == 2)
    checks.append(hilbert_fn([(2, 0)], 0, 2) == 1)
    # ideal (x^2, y^2) in k[x,y]: finite colength; deg 2 std monomials: {xy} only
    checks.append(hilbert_fn([(2, 0), (0, 2)], 2, 2) == 1)
    checks.append(hilbert_fn([(2, 0), (0, 2)], 3, 2) == 0)
    # colength of (x^2,y^2) = 4
    total = sum(hilbert_fn([(2, 0), (0, 2)], d, 2) for d in range(5))
    checks.append(total == 4)
    return float(sum(checks) / len(checks))


def bench_hilbert_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_series": _bench_hilbert_series(seed)}
