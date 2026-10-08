"""Tests for context-sensitive Andersen points-to analysis."""

from __future__ import annotations

from quant_fund.models.context_pta import CtxPTA, bench_context_pta


def test_return_flow_reaches_late_growing_callee() -> None:
    # The caller's only statement is a call: it visits once, reads the
    # callee's (still-empty) return points-to set, and is never re-queued
    # by its own facts. When the callee's return var gains objects later,
    # the caller must be re-visited or the call target misses them.
    a = CtxPTA(k=1)
    a.add_fn("mk", allocs=[("m", "o9")], ret="m")
    a.add_fn("h", calls=[("s", "mk", [], "u")])  # u = mk()
    a.add_fn("main", calls=[("c", "h", [], "p")])
    a.solve("main")
    assert a.points_to(("s",), "m") == {"&o9"}
    assert a.points_to(("c",), "u") == {"&o9"}


def test_formals_and_return_per_context() -> None:
    # Same callee under two call strings keeps facts split per context.
    a = CtxPTA(k=1)
    a.add_fn("id", formals=["a"], copies=[("r", "a")], ret="r")
    a.add_fn("f", allocs=[("x", "o1")], calls=[("s1", "id", ["x"], "u")])
    a.add_fn("g", allocs=[("y", "o2")], calls=[("s2", "id", ["y"], "v")])
    a.add_fn("main", calls=[("c1", "f", [], "p"), ("c2", "g", [], "q")])
    a.solve("main")
    assert a.points_to(("s1",), "a") == {"&o1"}
    assert a.points_to(("s2",), "a") == {"&o2"}
    # return values propagated back to each caller frame
    assert a.points_to(("c1",), "u") == {"&o1"}
    assert a.points_to(("c2",), "v") == {"&o2"}
    # monovariant view merges
    assert a.insensitive("a") == {"&o1", "&o2"}


def test_bench_context_pta() -> None:
    assert bench_context_pta()["synthetic_context_pta"] == 1.0
