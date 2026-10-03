from quant_fund.models.borel_subalgebra import bench_borel_subalgebra
from quant_fund.models.levi_factor import bench_levi_factor
from quant_fund.models.nilpotent_orbit import bench_nilpotent_orbit
from quant_fund.models.root_height import bench_root_height
from quant_fund.models.verma_module import bench_verma_module
from quant_fund.models.weyl_chamber import bench_weyl_chamber


def test_weyl_chamber():
    assert bench_weyl_chamber()["synthetic_weyl_chamber"] == 1.0


def test_root_height():
    assert bench_root_height()["synthetic_root_height"] == 1.0


def test_borel_subalgebra():
    assert bench_borel_subalgebra()["synthetic_borel_subalgebra"] == 1.0


def test_levi_factor():
    assert bench_levi_factor()["synthetic_levi_factor"] == 1.0


def test_nilpotent_orbit():
    assert bench_nilpotent_orbit()["synthetic_nilpotent_orbit"] == 1.0


def test_verma_module():
    assert bench_verma_module()["synthetic_verma_module"] == 1.0
