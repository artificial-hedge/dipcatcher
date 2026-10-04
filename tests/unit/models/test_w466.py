from quant_fund.models.kashiwara_schapira import bench_kashiwara_schapira
from quant_fund.models.loc_system import bench_loc_system
from quant_fund.models.micro_supp import bench_micro_supp
from quant_fund.models.perverse_2 import bench_perverse_2
from quant_fund.models.sheaf_homotopy import bench_sheaf_homotopy
from quant_fund.models.stacky_sheaf import bench_stacky_sheaf


def test_micro_supp():
    assert bench_micro_supp()["synthetic_micro_supp"] == 1.0


def test_kashiwara_schapira():
    assert bench_kashiwara_schapira()["synthetic_kashiwara_schapira"] == 1.0


def test_loc_system():
    assert bench_loc_system()["synthetic_loc_system"] == 1.0


def test_perverse_2():
    assert bench_perverse_2()["synthetic_perverse_2"] == 1.0


def test_stacky_sheaf():
    assert bench_stacky_sheaf()["synthetic_stacky_sheaf"] == 1.0


def test_sheaf_homotopy():
    assert bench_sheaf_homotopy()["synthetic_sheaf_homotopy"] == 1.0
