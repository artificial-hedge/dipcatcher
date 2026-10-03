from quant_fund.models.alexander_poly import bench_alexander_poly
from quant_fund.models.jones_poly import bench_jones_poly
from quant_fund.models.knot_group import bench_knot_group
from quant_fund.models.knot_invariant import bench_knot_invariant
from quant_fund.models.knot_signature import bench_knot_signature
from quant_fund.models.vassiliev_inv import bench_vassiliev_inv


def test_knot_invariant():
    assert bench_knot_invariant()["synthetic_knot_invariant"] == 1.0


def test_jones_poly():
    assert bench_jones_poly()["synthetic_jones_poly"] == 1.0


def test_alexander_poly():
    assert bench_alexander_poly()["synthetic_alexander_poly"] == 1.0


def test_knot_group():
    assert bench_knot_group()["synthetic_knot_group"] == 1.0


def test_knot_signature():
    assert bench_knot_signature()["synthetic_knot_signature"] == 1.0


def test_vassiliev_inv():
    assert bench_vassiliev_inv()["synthetic_vassiliev_inv"] == 1.0
