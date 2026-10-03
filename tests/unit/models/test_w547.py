from quant_fund.models.donaldson_thm import bench_donaldson_thm
from quant_fund.models.exotic_r4 import bench_exotic_r4
from quant_fund.models.four_mfd import bench_four_mfd
from quant_fund.models.freedman_thm import bench_freedman_thm
from quant_fund.models.intersection_form import bench_intersection_form
from quant_fund.models.seiberg_witten import bench_seiberg_witten


def test_four_mfd():
    assert bench_four_mfd()["synthetic_four_mfd"] == 1.0


def test_donaldson_thm():
    assert bench_donaldson_thm()["synthetic_donaldson_thm"] == 1.0


def test_seiberg_witten():
    assert bench_seiberg_witten()["synthetic_seiberg_witten"] == 1.0


def test_exotic_r4():
    assert bench_exotic_r4()["synthetic_exotic_r4"] == 1.0


def test_intersection_form():
    assert bench_intersection_form()["synthetic_intersection_form"] == 1.0


def test_freedman_thm():
    assert bench_freedman_thm()["synthetic_freedman_thm"] == 1.0
