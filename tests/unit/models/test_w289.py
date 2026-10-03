"""Unit tests for wave-289 category-theory canon modules."""

from quant_fund.models.adjunction import extend_set_map
from quant_fund.models.fin_cat import check
from quant_fund.models.functor_check import is_functor
from quant_fund.models.limit_prod import mediating
from quant_fund.models.monad_laws import _bind, _eta
from quant_fund.models.nat_trans import is_natural


def test_category_monoid():
    assert check(
        [0],
        {0: 0, 1: 0},
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        {0: 0},
    )


def test_functor_identity():
    assert is_functor(
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        {0: 0},
        {0: 0},
        {0: 0, 1: 2},
        {(a, b): (a + b) % 4 for a in range(4) for b in range(4)},
        {0: 0},
    )


def test_natural_identity():
    succ = {i: (i + 1) % 6 for i in range(6)}
    ident = {i: i for i in range(6)}
    assert is_natural(succ, succ, ident, [0])


def test_adjunction_ext():
    ext = extend_set_map({0: 1, 1: 2}, [0, 1], lambda a, b: (a + b) % 4, 0)
    assert ext((0, 1)) == 3 and ext(()) == 0


def test_mediating():
    h = mediating({0: 1, 1: 2}, {0: 0, 1: 1})
    assert h[0] == (1, 0) and h[1] == (2, 1)


def test_monad_bind():
    assert _bind([1, 2], lambda x: [x, x + 1]) == [1, 2, 2, 3]
    assert _bind(_eta(5), lambda x: [x * 2]) == [10]
