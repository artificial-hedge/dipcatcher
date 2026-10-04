from quant_fund.models.breuil_meizard import (
    bench_breuil_meizard,
)
from quant_fund.models.caruso_lebaron import (
    bench_caruso_lebaron,
)
from quant_fund.models.galdef_ring import bench_galdef_ring
from quant_fund.models.gee_kisin import bench_gee_kisin
from quant_fund.models.patching_arg import bench_patching_arg
from quant_fund.models.taylor_wiles import bench_taylor_wiles


def test_galdef_ring():
    assert bench_galdef_ring()["synthetic_galdef_ring"] == 1.0


def test_patching_arg():
    assert bench_patching_arg()["synthetic_patching_arg"] == 1.0


def test_taylor_wiles():
    assert bench_taylor_wiles()["synthetic_taylor_wiles"] == 1.0


def test_breuil_meizard():
    assert bench_breuil_meizard()["synthetic_breuil_meizard"] == 1.0


def test_gee_kisin():
    assert bench_gee_kisin()["synthetic_gee_kisin"] == 1.0


def test_caruso_lebaron():
    assert bench_caruso_lebaron()["synthetic_caruso_lebaron"] == 1.0
