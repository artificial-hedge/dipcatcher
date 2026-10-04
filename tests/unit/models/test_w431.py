from quant_fund.models.a1_homotopy import bench_a1_homotopy
from quant_fund.models.milnor_operations import (
    bench_milnor_operations,
)
from quant_fund.models.morel_degree import bench_morel_degree
from quant_fund.models.motivic_sphere import bench_motivic_sphere
from quant_fund.models.slice_filtration import (
    bench_slice_filtration,
)
from quant_fund.models.voevodsky_motive import (
    bench_voevodsky_motive,
)


def test_a1_homotopy():
    assert bench_a1_homotopy()["synthetic_a1_homotopy"] == 1.0


def test_motivic_sphere():
    assert bench_motivic_sphere()["synthetic_motivic_sphere"] == 1.0


def test_morel_degree():
    assert bench_morel_degree()["synthetic_morel_degree"] == 1.0


def test_voevodsky_motive():
    assert bench_voevodsky_motive()["synthetic_voevodsky_motive"] == 1.0


def test_slice_filtration():
    assert bench_slice_filtration()["synthetic_slice_filtration"] == 1.0


def test_milnor_operations():
    assert bench_milnor_operations()["synthetic_milnor_operations"] == 1.0
