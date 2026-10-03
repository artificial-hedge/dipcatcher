"""Wave-322 verification-2 module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.cegis_loop import cegis
from quant_fund.models.horn_clauses import saturate
from quant_fund.models.interpolant_mc import interpolate_bmc
from quant_fund.models.predicate_abs import reach_abstract
from quant_fund.models.sygus_synth import _eval as seval
from quant_fund.models.sygus_synth import synth
from quant_fund.models.weakest_precond import _eval, verify, wp


def test_wp_assign() -> None:
    f = wp(("assign", "x", ("add", ("var", "x"), ("lit", 1))), ("le", ("var", "x"), ("lit", 5)))
    assert _eval(f, {"x": 4}) is True
    assert _eval(f, {"x": 5}) is False


def test_wp_swap() -> None:
    prog = (
        "seq",
        ("assign", "t", ("var", "x")),
        ("seq", ("assign", "x", ("var", "y")), ("assign", "y", ("var", "t"))),
    )
    assert verify(
        ("and", ("eq", ("var", "x"), ("var", "x0")), ("eq", ("var", "y"), ("var", "y0"))),
        prog,
        ("and", ("eq", ("var", "x"), ("var", "y0")), ("eq", ("var", "y"), ("var", "x0"))),
        {v: range(3) for v in ("x", "y", "x0", "y0", "t")},
    )


def test_sygus_max() -> None:
    ios = [{"x": 1, "y": 2, "out": 2}, {"x": 3, "y": 0, "out": 3}]
    spec = lambda out, e: out >= e["x"] and out >= e["y"] and out in (e["x"], e["y"])  # noqa: E731
    t = synth(spec, ios, ["x", "y"], depth=2)
    assert t is not None and seval(t, {"x": -4, "y": 2}) == 2


def test_horn_sat() -> None:
    assert 2 in saturate(set(), [([], 0), ([0], 1), ([1], 2)])


def test_interp_safe() -> None:
    safe, _ = interpolate_bmc({0}, lambda x: min(x + 1, 4), lambda x: x <= 4, 6)
    assert safe


def test_pred_abs() -> None:
    preds = [lambda x: x < 0]
    reach = reach_abstract({2}, lambda x: max(x - 1, 0), preds, range(-2, 6))
    assert (True,) not in reach


def test_cegis() -> None:
    spec = lambda out, e: out >= e["x"] and out >= e["y"] and out in (e["x"], e["y"])  # noqa: E731
    t, it = cegis(spec, ["x", "y"], range(-2, 3))
    assert t is not None and it <= 12


def test_wp_bad() -> None:
    with pytest.raises(ValueError):
        wp(("bogus",), ("lit", 1))
