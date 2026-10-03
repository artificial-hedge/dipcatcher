"""Unit tests for wave-302 compiler-5/JIT canon modules."""

from quant_fund.models.card_table_gc import Heap
from quant_fund.models.escape_analysis import escapes, scalar_replace
from quant_fund.models.gvn_pre import anticipatable, count_evals, pre_insert
from quant_fund.models.osr_deopt import OptimizedFrame, run_baseline
from quant_fund.models.ssa_repair import exec_copies, schedule_copies
from quant_fund.models.trace_tree import simulate


def test_card_table_remembered():
    h = Heap(1024, 64)
    h.write_old(5, 3)
    h.write_old(6, 4)
    h.write_old(900, 7)
    assert h.remember() == [3, 4, 7]
    assert h.remember() == []  # cards cleared after scan


def test_escape_basic():
    prog = [
        ("alloc", "p", "1", "2"),
        ("field", "x", "p", "0"),
        ("alloc", "q", "9", "x"),
        ("return", "q"),
    ]
    assert escapes(prog) == {"q"}
    out = scalar_replace(prog)
    assert sum(1 for op in out if op[0] == "alloc") == 1
    assert ("const", "x", "1") in out


def test_osr_exact():
    prog = [("add", "a", "b", 1), ("mul", "c", "a", 2)]
    oracle = run_baseline(prog, 0, {"b": 3})
    fr = OptimizedFrame(prog, 0)
    fr.locals = {"b": 3}
    assert fr.deopt() == oracle


def test_trace_accounting():
    its = ["a"] * 50 + ["b"] * 10
    res = simulate(4, its)
    assert res["on_trace"] + res["side_exit"] == 60
    assert res["side_traces"] == 1


def test_ssa_swap():
    seq = schedule_copies([("x", "y"), ("y", "x")])
    env = exec_copies({"x": 1, "y": 2, "_scratch": 0}, seq)
    assert env["x"] == 2 and env["y"] == 1


def test_gvn_pre_diamond():
    cfg = {
        "entry": {"succ": ["L", "R"], "exprs": [], "kill": []},
        "L": {"succ": ["join"], "exprs": [("x+y", "t")], "kill": []},
        "R": {"succ": ["join"], "exprs": [], "kill": []},
        "join": {"succ": [], "exprs": [("x+y", "u")], "kill": []},
    }
    assert anticipatable(cfg, "entry", "x+y") == {"entry", "L", "R"}
    out = pre_insert(cfg, "entry", "x+y")
    assert any(e == "x+y" for e, _ in out["R"]["exprs"])
    assert count_evals(out, "x+y") >= 3
