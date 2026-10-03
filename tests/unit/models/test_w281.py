"""Wave-281 abstract-algebra module tests."""

from quant_fund.models.group_table import _klein4, is_group
from quant_fund.models.ideal_member import reduce_poly
from quant_fund.models.perm_group import compose, order, sign
from quant_fund.models.poly_ring import pdiv


def test_klein4_valid() -> None:
    ok, e = is_group(_klein4())
    assert ok and e == 0


def test_perm_order_transposition() -> None:
    assert order([1, 0, 2]) == 2


def test_perm_sign() -> None:
    assert sign([1, 0, 2]) == -1
    assert sign(compose([1, 0, 2], [1, 0, 2])) == 1


def test_poly_div_exact() -> None:
    q, r = pdiv([1, 2, 1], [1, 1], 7)  # (x+1)^2 / (x+1)
    assert r == [0] and q == [1, 1]


def test_ideal_reduce() -> None:
    rem = reduce_poly([(1, 2, 0), (1, 0, 1)], [(1, 1, 0)])
    assert rem == [(1, 0, 1)]
