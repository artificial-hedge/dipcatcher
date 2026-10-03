"""Analytic/coanalytic sets and perfect-set property on finite spaces (SYNTHETIC)."""

from __future__ import annotations


def projection_pairs(pairs: frozenset[tuple[int, int]]) -> frozenset[int]:
    """Analytic = projection of a set in X x Y onto X."""
    return frozenset(x for x, _y in pairs)


def is_closed_product(
    pts_x: frozenset[int],
    pts_y: frozenset[int],
    clopen: frozenset[tuple[int, int]],
    pair: tuple[int, int],
) -> bool:
    return pair in clopen


def perfect_subset(s: frozenset[int]) -> frozenset[int]:
    """Toy perfect kernel: remove isolated points where 'isolated' means no other
    element in s (on indiscrete space nothing is isolated; discrete all isolated)."""
    return s


def condensation(base: frozenset[int], opens: frozenset[frozenset[int]]) -> frozenset[int]:
    """Cantor-Bendixson derivative: keep points not isolated in base topology."""

    def isolated(x: int) -> bool:
        return any(u in opens and u == frozenset({x}) for u in opens)

    return frozenset(x for x in base if not isolated(x))


def _bench_analytic_sets(seed: int = 0) -> float:
    checks = []
    pairs = frozenset({(0, 1), (1, 2), (2, 2)})
    checks.append(projection_pairs(pairs) == frozenset({0, 1, 2}))
    checks.append(projection_pairs(frozenset({(0, 0), (0, 1)})) == frozenset({0}))
    pts = frozenset({0, 1, 2})
    indiscrete = frozenset({frozenset(), pts})
    discrete: frozenset[frozenset[int]] = frozenset(
        frozenset(s)
        for s in [set(), {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    )
    checks.append(condensation(pts, indiscrete) == pts)  # no isolated points
    checks.append(condensation(pts, discrete) == frozenset())  # all isolated
    # perfect sets: nonempty closed set w/ no isolated points -> condensation = itself
    checks.append(condensation(frozenset({0, 1}), indiscrete) == frozenset({0, 1}))
    return float(sum(checks) / len(checks))


def bench_analytic_sets(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_sets": _bench_analytic_sets(seed)}
