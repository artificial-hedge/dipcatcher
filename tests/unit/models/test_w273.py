"""Wave-273 compiler-3 module tests."""

import numpy as np

from quant_fund.models.const_fold import fold
from quant_fund.models.inline_expand import inline_call, run_expr
from quant_fund.models.loop_unroll import unrolled_sum
from quant_fund.models.partial_eval import peval
from quant_fund.models.peephole_opt import peephole, run_prog
from quant_fund.models.strength_red import eval_ops, reduce_mul


def test_peval_folds_known() -> None:
    e = ("mul", ("var", "x"), ("const", 0))
    assert peval(e, {"x": 5}) == ("const", 0)


def test_peval_leaves_unknown() -> None:
    e = ("add", ("var", "y"), ("const", 1))
    assert peval(e, {"x": 5}) == e


def test_peephole_const_fold() -> None:
    prog = [("push", 2), ("push", 3), "add"]
    assert run_prog(peephole(prog), {}) == 5


def test_peephole_neg_neg() -> None:
    prog = [("push", 7), "neg", "neg"]
    assert run_prog(peephole(prog), {}) == 7


def test_strength_red_known() -> None:
    assert eval_ops(reduce_mul(5), 7) == 35
    assert eval_ops(reduce_mul(-3), 4) == -12


def test_const_fold_chain() -> None:
    stmts = [("a", "const", 2), ("b", "add", "a", 3), ("c", "mul", "a", "b")]
    env = fold(stmts)
    assert env["c"] == 10


def test_unroll_matches_sum() -> None:
    arr = np.arange(13.0)
    for u in (2, 3, 4, 8):
        assert unrolled_sum(arr, u) == arr.sum()


def test_inline_call() -> None:
    f = ("a", ("add", ("var", "a"), ("const", 1)))
    body = inline_call(f, ("var", "x"))
    assert run_expr(body, {"x": 4}) == 5
