from quant_fund.models.g_spectrum import bench_g_spectrum
from quant_fund.models.mackey_functor import bench_mackey_functor
from quant_fund.models.norm_map import bench_norm_map
from quant_fund.models.ro_grading import bench_ro_grading
from quant_fund.models.tom_dieck import bench_tom_dieck
from quant_fund.models.wirthmuller import bench_wirthmuller


def test_g_spectrum():
    assert bench_g_spectrum()["synthetic_g_spectrum"] == 1.0


def test_mackey_functor():
    assert bench_mackey_functor()["synthetic_mackey_functor"] == 1.0


def test_norm_map():
    assert bench_norm_map()["synthetic_norm_map"] == 1.0


def test_ro_grading():
    assert bench_ro_grading()["synthetic_ro_grading"] == 1.0


def test_wirthmuller():
    assert bench_wirthmuller()["synthetic_wirthmuller"] == 1.0


def test_tom_dieck():
    assert bench_tom_dieck()["synthetic_tom_dieck"] == 1.0
