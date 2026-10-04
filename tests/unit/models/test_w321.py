"""Wave-321 shape-analysis module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.context_pta import CtxPTA
from quant_fund.models.interproc_summary import _exec_fn, _summary, apply_summary
from quant_fund.models.recency_abstraction import RecencyHeap
from quant_fund.models.separation_logic import _holds, entails
from quant_fund.models.shape_graph import canonical_embed, garbage, is_list
from quant_fund.models.three_valued_logic import U, eval_formula, tc_struct


def test_tvla_tc() -> None:
    st = {
        "inds": [0, 1, 2],
        "preds": {"nxt": {(0, 1): 1.0, (1, 2): 1.0}},
    }
    st["preds"]["reach"] = tc_struct(st)
    assert st["preds"]["reach"][(0, 2)] == 1.0
    assert eval_formula(("const", U), st) == U


def test_shape_merge() -> None:
    heap = {0: {"nxt": 1}, 1: {"nxt": 2}, 2: {"nxt": None}}
    nodes, _ = canonical_embed(heap, [0])
    assert len(nodes) == 1
    assert is_list(heap, 0)
    assert garbage(heap, [0]) == set()


def test_seplog_list() -> None:
    env = {"x": 1, "y": 2}
    assert _holds(("list", "x"), env, {1: 2, 2: None})
    f = ("star", ("mapsto", "x", "y"), ("mapsto", "y", "null"))
    assert entails(f, ("list", "x"), [1, 2], env)


def test_context_k1() -> None:
    a = CtxPTA(k=1)
    a.add_fn("id", formals=["a"], copies=[("r", "a")], ret="r")
    a.add_fn("f", allocs=[("x", "o1")], calls=[("s1", "id", ["x"], "u")])
    a.add_fn("main", calls=[("c1", "f", [], "p")])
    a.solve("main")
    assert a.points_to(("s1",), "a") == {"&o1"}


def test_recency() -> None:
    h = RecencyHeap()
    h.alloc()
    h.set_field("val", 5)
    h.alloc()
    assert h.fresh_val_range() == (-1, -1)
    assert h.old_val_bounds() == (5, 5)


def test_summary_apply() -> None:
    prog = {"dbl": {"body": [("copy", "r2", "x")]}}
    s = _summary(prog, "dbl", ["x"], ["r2"])
    assert apply_summary(s, (2,)) == (2,)
    assert _exec_fn(prog, "dbl", {"x": 1})["r2"] == 1


def test_tvla_bad_formula() -> None:
    with pytest.raises(ValueError):
        eval_formula(("bogus",), {"inds": [], "preds": {}})
