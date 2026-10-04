from quant_fund.models.analytic_ring import bench_analytic_ring
from quant_fund.models.condensed_set import bench_condensed_set
from quant_fund.models.light_condensed import bench_light_condensed
from quant_fund.models.liquid_group import bench_liquid_group
from quant_fund.models.proetale_site import bench_proetale_site
from quant_fund.models.solid_group import bench_solid_group


def test_condensed_set():
    assert bench_condensed_set()["synthetic_condensed_set"] == 1.0


def test_solid_group():
    assert bench_solid_group()["synthetic_solid_group"] == 1.0


def test_liquid_group():
    assert bench_liquid_group()["synthetic_liquid_group"] == 1.0


def test_proetale_site():
    assert bench_proetale_site()["synthetic_proetale_site"] == 1.0


def test_light_condensed():
    assert bench_light_condensed()["synthetic_light_condensed"] == 1.0


def test_analytic_ring():
    assert bench_analytic_ring()["synthetic_analytic_ring"] == 1.0
