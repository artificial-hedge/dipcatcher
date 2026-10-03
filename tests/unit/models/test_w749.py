from quant_fund.models.borodin_bufetov import (
    bench_borodin_bufetov,
)
from quant_fund.models.borodin_wheeler import (
    bench_borodin_wheeler,
)
from quant_fund.models.bufetov_sixv import bench_bufetov_sixv
from quant_fund.models.dimitrov_sixv import bench_dimitrov_sixv
from quant_fund.models.kuan_sixv import bench_kuan_sixv
from quant_fund.models.wheeler_zinn import bench_wheeler_zinn


def test_bufetov_sixv():
    assert bench_bufetov_sixv()["synthetic_bufetov_sixv"] == 1.0


def test_borodin_bufetov():
    assert bench_borodin_bufetov()["synthetic_borodin_bufetov"] == 1.0


def test_kuan_sixv():
    assert bench_kuan_sixv()["synthetic_kuan_sixv"] == 1.0


def test_dimitrov_sixv():
    assert bench_dimitrov_sixv()["synthetic_dimitrov_sixv"] == 1.0


def test_borodin_wheeler():
    assert bench_borodin_wheeler()["synthetic_borodin_wheeler"] == 1.0


def test_wheeler_zinn():
    assert bench_wheeler_zinn()["synthetic_wheeler_zinn"] == 1.0
