"""CFG simplification edge tests."""

from __future__ import annotations

from quant_fund.models.cfg_simplify import bench_cfg_simplify, simplify


def test_jmp_to_entry_does_not_delete_entry():
    """A back-edge jmp to the entry block used to merge entry away,
    returning a CFG with no entry point."""
    cfg = {
        "e": ("br", "c", "x", "a"),
        "a": ("jmp", "e"),
        "x": ("ret", "z"),
        "c": ("ret", "w"),
    }
    s = simplify(cfg, "e")
    assert "e" in s
    assert s["a"] == ("jmp", "e")


def test_unreachable_dropped():
    cfg = {"e": ("jmp", "a"), "a": ("ret", "z"), "dead": ("jmp", "a")}
    assert simplify(cfg, "e") == {"e": ("ret", "z")}


def test_shared_succ_not_merged():
    cfg = {
        "a": ("jmp", "t"),
        "b": ("jmp", "t"),
        "t": ("ret", "z"),
        "e": ("br", "c", "a", "b"),
    }
    assert len(simplify(cfg, "e")) == 4


def test_bench():
    assert bench_cfg_simplify()["synthetic_cfg"] == 1.0
