from quant_fund.models.bn_pair import bench_bn_pair
from quant_fund.models.braid_grp import bench_braid_grp
from quant_fund.models.building_toy import bench_building_toy
from quant_fund.models.coxeter_grp import bench_coxeter_grp
from quant_fund.models.hecke_bm import bench_hecke_bm
from quant_fund.models.parabolic_grp import bench_parabolic_grp


def test_building_toy():
    assert bench_building_toy()["synthetic_building_toy"] == 1.0


def test_coxeter_grp():
    assert bench_coxeter_grp()["synthetic_coxeter_grp"] == 1.0


def test_bn_pair():
    assert bench_bn_pair()["synthetic_bn_pair"] == 1.0


def test_braid_grp():
    assert bench_braid_grp()["synthetic_braid_grp"] == 1.0


def test_hecke_bm():
    assert bench_hecke_bm()["synthetic_hecke_bm"] == 1.0


def test_parabolic_grp():
    assert bench_parabolic_grp()["synthetic_parabolic_grp"] == 1.0
