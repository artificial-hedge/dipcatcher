from quant_fund.models.adams_ss import bench_adams_ss
from quant_fund.models.cofiber import bench_cofiber
from quant_fund.models.exact_couple import bench_exact_couple
from quant_fund.models.obstruction import bench_obstruction
from quant_fund.models.stable_homotopy import bench_stable_homotopy
from quant_fund.models.whitehead_thm import bench_whitehead_thm


def test_exact_couple():
    assert bench_exact_couple()["synthetic_exact_couple"] == 1.0


def test_adams_ss():
    assert bench_adams_ss()["synthetic_adams_ss"] == 1.0


def test_stable_homotopy():
    assert bench_stable_homotopy()["synthetic_stable_homotopy"] == 1.0


def test_whitehead_thm():
    assert bench_whitehead_thm()["synthetic_whitehead_thm"] == 1.0


def test_obstruction():
    assert bench_obstruction()["synthetic_obstruction"] == 1.0


def test_cofiber():
    assert bench_cofiber()["synthetic_cofiber"] == 1.0
