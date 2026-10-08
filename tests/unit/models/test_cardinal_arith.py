from quant_fund.models.cardinal_arith import (
    bench_cardinal_arith,
    bijection,
    cantor,
    card_prod,
    card_sum,
    embeds,
    power_set,
)


def test_bijection_is_equal_cardinality() -> None:
    # finite sets admit a bijection iff |a| == |b| — elements are
    # irrelevant; the check must not pretend to inspect them.
    assert bijection(frozenset({1, 2, 3}), frozenset({"p", "q", "r"}))
    assert bijection(frozenset(), frozenset())
    assert not bijection(frozenset({1, 2}), frozenset({1, 2, 3}))
    assert not bijection(frozenset({1, 2, 3}), frozenset())


def test_cantor_strict_dominance() -> None:
    a = frozenset({1, 2, 3})
    assert len(power_set(a)) == 8
    assert cantor(a)
    assert cantor(frozenset())


def test_arithmetic_laws() -> None:
    a, b = frozenset({1, 2, 3}), frozenset({"x", "y"})
    assert len(card_sum(a, b)) == 5
    assert len(card_prod(a, b)) == 6
    assert embeds(b, a) and not embeds(a, b)


def test_bench_passes() -> None:
    assert bench_cardinal_arith()["synthetic_cardinal_arith"] == 1.0
