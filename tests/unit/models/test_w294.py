"""Unit tests for wave-294 compiler-4 canon modules."""

from quant_fund.models.bb_reorder import layout
from quant_fund.models.cfg_simplify import simplify
from quant_fund.models.jump_thread import thread
from quant_fund.models.modulo_sched import ii
from quant_fund.models.tail_dup import dup, trace
from quant_fund.models.tree_cover import opt_cover


def test_cover_mac():
    a, b, c = ("leaf", "a"), ("leaf", "b"), ("leaf", "c")
    assert opt_cover(("*", ("+", a, b), c), {}) == (4, ["MAC"])


def test_ii():
    assert ii(["load", "fmul", "fadd", "store"], [(1, 1)]) == 2
    assert ii(["alu"], [(3, 1)]) == 3


def test_thread():
    cfg = {
        "entry": ("br", "x>0", "A", "Z"),
        "A": ("br", "y>0", "T", "Z"),
        "T": ("ret", "hot"),
        "Z": ("ret", "cold"),
    }
    assert thread(cfg, {"x>0": True, "y>0": True})[-1] == "hot"
    assert thread(cfg, {"x>0": False})[-1] == "cold"


def test_tail_dup():
    cfg = {
        "b1": ("br", "c1", "b2", "b3"),
        "b2": ("jmp", "b4"),
        "b3": ("jmp", "b4"),
        "b4": ("br", "c2", "b5", "b6"),
        "b5": ("ret", "x"),
        "b6": ("ret", "y"),
    }
    ncfg = dup(cfg, "b1", ("b2", "b4"))
    assert "b4__dup_b2" in ncfg
    assert trace(ncfg, "b1", {"c1": True, "c2": True})[-1] == "b5"


def test_simplify():
    cfg = {"e": ("jmp", "a"), "a": ("jmp", "b"), "b": ("ret", "z")}
    assert simplify(cfg, "e") == {"e": ("ret", "z")}


def test_reorder():
    cfg = {
        "e": ("br", "c", "a", "z"),
        "a": ("jmp", "m"),
        "m": ("jmp", "x"),
        "z": ("jmp", "x"),
        "x": ("ret", "r"),
    }
    o = layout(cfg, "e", {("e", "a"): 90, ("e", "z"): 10})
    assert o.index("a") == o.index("e") + 1
