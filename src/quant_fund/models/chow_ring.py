"""Chow ring computations on projective spaces (SYNTHETIC)."""

from __future__ import annotations


def chow_pn_deg(cls: tuple[int, int], n: int) -> int:
    """A*(P^n) = Z[h]/(h^{n+1}); the class h^c has degree 1 when c = n.

    cls = (power of h, coefficient); product with another class adds
    powers mod n+1; degree of h^n is 1.
    """
    power, coeff = cls
    if power == n:
        return coeff
    return 0


def chow_mul(a: tuple[int, int], b: tuple[int, int], n: int) -> tuple[int, int]:
    """Multiply two classes (h^p coeff) in A*(P^n); result is zero if
    p + q > n."""
    p, ca = a
    q, cb = b
    if p + q > n:
        return (p + q, 0)
    return (p + q, ca * cb)


def bezout_pn(degs: list[int], n: int) -> int:
    """Bezout on P^n: n hypersurfaces of degrees d_i meet in prod d_i points."""
    out = 1
    for d in degs:
        out *= d
    return out


def _bench_chow_ring(seed: int = 0) -> float:
    checks = []
    # A*(P2): h^2 degree 1
    checks.append(chow_pn_deg((2, 1), 2) == 1)
    checks.append(chow_pn_deg((1, 5), 2) == 0)
    # h * h = h^2 on P2
    checks.append(chow_mul((1, 1), (1, 1), 2) == (2, 1))
    # h^2 * h = 0 on P2
    checks.append(chow_mul((2, 1), (1, 1), 2) == (3, 0))
    # two lines in P2 meet in 1 point
    checks.append(bezout_pn([1, 1], 2) == 1)
    # conic x cubic in P2: 6 points
    checks.append(bezout_pn([2, 3], 2) == 6)
    # three planes in P3: 1 point; quadric x plane: 2
    checks.append(bezout_pn([1, 1, 1], 3) == 1)
    checks.append(bezout_pn([2, 1, 1], 3) == 2)
    return float(sum(checks) / len(checks))


def bench_chow_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chow_ring": _bench_chow_ring(seed)}
