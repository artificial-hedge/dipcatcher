from quant_fund.models.borel_hierarchy import (
    bench_borel_hierarchy,
    complements,
    intersections_all,
    sigma_rank,
    unions_all,
)


def test_intersections_all_keeps_every_subfamily():
    """Probe for the dropped-acc bug: {A,B,C} must yield A&B, A&C, B&C
    and A&B&C — not only the chains ending at the last element."""
    a, b, c = frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})
    out = intersections_all(frozenset({a, b, c}))
    assert a & b in out  # {1}
    assert a & c in out  # {}
    assert b & c in out  # {2}
    assert a & b & c in out
    assert a in out and b in out and c in out


def test_intersections_all_singleton_and_empty():
    a = frozenset({0})
    assert intersections_all(frozenset({a})) == frozenset({a})
    assert intersections_all(frozenset()) == frozenset()


def test_sigma_rank_levels():
    pts = frozenset({0, 1, 2})
    opens = frozenset({frozenset(), frozenset({0}), frozenset({0, 1}), pts})
    assert sigma_rank(pts, opens, frozenset({0})) == 0
    assert sigma_rank(pts, opens, frozenset({2})) == 1


def test_complements_and_unions():
    pts = frozenset({0, 1})
    fam = frozenset({frozenset({0}), frozenset({1})})
    assert complements(pts, fam) == frozenset({frozenset({1}), frozenset({0})})
    assert frozenset({0, 1}) in unions_all(fam)


def test_bench_perfect():
    assert bench_borel_hierarchy()["synthetic_borel_hierarchy"] == 1.0
