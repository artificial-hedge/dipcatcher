from quant_fund.models.closed_cat import bench_closed_cat
from quant_fund.models.distributor import bench_distributor
from quant_fund.models.equivalence_cat import bench_equivalence_cat
from quant_fund.models.kan_extension import bench_kan_extension
from quant_fund.models.monoidal_cat import bench_monoidal_cat
from quant_fund.models.presheaf import bench_presheaf


def test_monoidal_cat():
    assert bench_monoidal_cat()["synthetic_monoidal_cat"] == 1.0


def test_closed_cat():
    assert bench_closed_cat()["synthetic_closed_cat"] == 1.0


def test_presheaf():
    assert bench_presheaf()["synthetic_presheaf"] == 1.0


def test_kan_extension():
    assert bench_kan_extension()["synthetic_kan_extension"] == 1.0


def test_distributor():
    assert bench_distributor()["synthetic_distributor"] == 1.0


def test_equivalence_cat():
    assert bench_equivalence_cat()["synthetic_equivalence_cat"] == 1.0
