from quant_fund.models.campbell_thm import bench_campbell_thm
from quant_fund.models.cox_process import bench_cox_process
from quant_fund.models.hawkes_point import bench_hawkes_point
from quant_fund.models.marked_point import bench_marked_point
from quant_fund.models.palm_dist import bench_palm_dist
from quant_fund.models.self_excite import bench_self_excite


def test_cox_process():
    assert bench_cox_process()["synthetic_cox_process"] == 1.0


def test_hawkes_point():
    assert bench_hawkes_point()["synthetic_hawkes_point"] == 1.0


def test_self_excite():
    assert bench_self_excite()["synthetic_self_excite"] == 1.0


def test_marked_point():
    assert bench_marked_point()["synthetic_marked_point"] == 1.0


def test_campbell_thm():
    assert bench_campbell_thm()["synthetic_campbell_thm"] == 1.0


def test_palm_dist():
    assert bench_palm_dist()["synthetic_palm_dist"] == 1.0
