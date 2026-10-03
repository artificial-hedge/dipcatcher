from quant_fund.models.chern_character import bench_chern_character
from quant_fund.models.chern_class import bench_chern_class
from quant_fund.models.euler_class import bench_euler_class
from quant_fund.models.hirzebruch_sig import bench_hirzebruch_sig
from quant_fund.models.pontryagin_class import bench_pontryagin_class
from quant_fund.models.todd_genus import bench_todd_genus


def test_chern_class():
    assert bench_chern_class()["synthetic_chern_class"] == 1.0


def test_pontryagin_class():
    assert bench_pontryagin_class()["synthetic_pontryagin_class"] == 1.0


def test_euler_class():
    assert bench_euler_class()["synthetic_euler_class"] == 1.0


def test_todd_genus():
    assert bench_todd_genus()["synthetic_todd_genus"] == 1.0


def test_chern_character():
    assert bench_chern_character()["synthetic_chern_character"] == 1.0


def test_hirzebruch_sig():
    assert bench_hirzebruch_sig()["synthetic_hirzebruch_sig"] == 1.0
