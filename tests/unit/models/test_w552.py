from quant_fund.models.contact_form import bench_contact_form
from quant_fund.models.convex_surface import bench_convex_surface
from quant_fund.models.giroux_corr import bench_giroux_corr
from quant_fund.models.legendrian_knot import bench_legendrian_knot
from quant_fund.models.overtwisted import bench_overtwisted
from quant_fund.models.tight_contact import bench_tight_contact


def test_contact_form():
    assert bench_contact_form()["synthetic_contact_form"] == 1.0


def test_legendrian_knot():
    assert bench_legendrian_knot()["synthetic_legendrian_knot"] == 1.0


def test_overtwisted():
    assert bench_overtwisted()["synthetic_overtwisted"] == 1.0


def test_tight_contact():
    assert bench_tight_contact()["synthetic_tight_contact"] == 1.0


def test_giroux_corr():
    assert bench_giroux_corr()["synthetic_giroux_corr"] == 1.0


def test_convex_surface():
    assert bench_convex_surface()["synthetic_convex_surface"] == 1.0
