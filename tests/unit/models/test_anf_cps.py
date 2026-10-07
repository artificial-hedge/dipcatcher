"""Tests for models/anf_cps.py — ANF conversion preserves eval and
never captures existing variable names."""

from __future__ import annotations

import numpy as np

from quant_fund.models.anf_cps import _eval, _is_anf, anf, bench_anf_cps


def test_anf_atomic_noop() -> None:
    assert anf(("lit", 3.0)) == ("lit", 3.0)
    assert anf(("var", "x")) == ("var", "x")


def test_anf_preserves_eval() -> None:
    t = ("add", ("mul", ("var", "x"), ("lit", 2.0)), ("let", "z", ("lit", 1.5), ("var", "z")))
    a = anf(t)
    env = {"x": 3.0, "z": 0.0}
    assert np.isclose(_eval(a, env), _eval(t, env))
    assert _is_anf(a)


def test_anf_no_capture_of_used_names() -> None:
    """A term already containing `_t0` must not have that name captured
    by a hoisted let — capture flips eval from 16 to 12."""
    t = ("add", ("mul", ("lit", 2.0), ("lit", 3.0)), ("var", "_t0"))
    a = anf(t)
    env = {"_t0": 10.0}
    assert np.isclose(_eval(a, env), _eval(t, env))
    # the hoist binder must be fresh (_t0 is taken)
    assert a[0] == "let" and a[1] != "_t0"


def test_anf_let_binder_not_shadowed() -> None:
    """A hoist inside a let-body must not collide with the let binder:
    let _t0 = 5 in (mul _t0) uses _t0=5, not the hoisted product."""
    t = ("let", "_t0", ("lit", 5.0), ("add", ("mul", ("lit", 2.0), ("lit", 3.0)), ("var", "_t0")))
    a = anf(t)
    assert np.isclose(_eval(a, {}), _eval(t, {}))  # 11.0, not 12.0


def test_bench_anf_cps() -> None:
    out = bench_anf_cps()
    assert out["synthetic_eval_preserved"] == 1.0
    assert out["synthetic_anf_form"] == 1.0
