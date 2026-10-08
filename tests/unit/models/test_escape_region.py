"""Escape-region inference honesty tests."""

from __future__ import annotations

from quant_fund.models.escape_region import bench_escape_region, infer


def test_infer_marks_escaping_region():
    prog = ("letregion", "r", ("at", ("lit", 1), "r"))
    assert infer(prog) == {"r0:r": "escape"}


def test_infer_marks_confined_region():
    prog = ("letregion", "r", ("let", "t", ("at", ("lit", 1), "r"), ("lit", 5)))
    assert infer(prog) == {"r0:r": "stack"}


def test_infer_letbound_var_escape():
    prog = ("letregion", "r", ("let", "x", ("at", ("lit", 1), "r"), ("var", "x")))
    assert infer(prog) == {"r0:r": "escape"}


def test_infer_unbound_region_escapes():
    prog = ("at", ("lit", 9), "z")
    assert infer(prog)["r0:z"] == "escape"


def test_bench_escape_region():
    assert bench_escape_region()["synthetic_escape_region"] == 1.0
