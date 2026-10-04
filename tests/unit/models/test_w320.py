"""Wave-320 abstract-interpretation module unit tests."""

from __future__ import annotations

from fractions import Fraction

import pytest

from quant_fund.models.affine_karr import AffineState
from quant_fund.models.andersen_pta import Andersen
from quant_fund.models.chaotic_widen import TOP, lfp
from quant_fund.models.interval_analysis import INF, run_concrete, transfer
from quant_fund.models.sign_domain import NEG, POS, _abs_eval
from quant_fund.models.sign_domain import TOP as STOP
from quant_fund.models.zone_dbm import Zone


def test_interval_loop() -> None:
    x = ("var", "x")
    prog = (
        "seq",
        ("assign", "x", ("lit", 0.0)),
        ("while", ("lt", x, ("lit", 10.0)), ("assign", "x", ("add", x, ("lit", 1.0)))),
    )
    s = transfer(prog, {"x": (-INF, INF)})
    assert s["x"] == (10.0, 10.0)
    assert run_concrete(prog, {"x": 0.0})["x"] == 10.0


def test_sign_mul() -> None:
    assert _abs_eval(("mul", ("var", "a"), ("var", "a")), {"a": NEG}) == POS
    assert _abs_eval(("add", ("var", "a"), ("neg", ("var", "a"))), {"a": STOP}) == STOP


def test_zone_closure() -> None:
    z = Zone(2)
    z.add_upper(1, 5.0)
    z.add_diff(1, 2, 1.0)
    z.add_upper(2, 3.0)
    assert z.bound(1)[1] == 4.0
    z.add_lower(1, 6.0)
    assert z.is_empty()


def test_karr_assign() -> None:
    s = AffineState.top(2)
    s.assign(0, [Fraction(0), Fraction(0)], Fraction(3))
    s.assign(1, [Fraction(2), Fraction(0)], Fraction(0))
    assert s.implies([Fraction(0), Fraction(1)], Fraction(6))


def test_widen_lfp() -> None:
    f = lambda v: (0 if v == -1 else min(v + 1, 150)) if v != TOP else TOP  # noqa: E731
    val, _ = lfp(f, 200, widen_at=2)
    assert val == 150


def test_andersen_flow_ins() -> None:
    a = Andersen()
    a.add_addr("p", "x")
    a.add_copy("q", "p")
    a.add_load("r", "q")
    a.add_addr("s", "y")
    a.add_store("p", "s")
    a.solve()
    assert "&y" in a.points_to("r")
    assert a.alias("r", "p")


def test_interval_unknown_op() -> None:
    with pytest.raises(ValueError):
        transfer(("noop",), {})
