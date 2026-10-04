from quant_fund.models.motivic_euler import bench_motivic_euler
from quant_fund.models.motivic_ext import bench_motivic_ext
from quant_fund.models.motivic_infinite2 import (
    bench_motivic_infinite2,
)
from quant_fund.models.motivic_jouanolou import (
    bench_motivic_jouanolou,
)
from quant_fund.models.motivic_norm import bench_motivic_norm
from quant_fund.models.motivic_ramified import (
    bench_motivic_ramified,
)


def test_motivic_jouanolou():
    assert bench_motivic_jouanolou()["synthetic_motivic_jouanolou"] == 1.0


def test_motivic_infinite2():
    assert bench_motivic_infinite2()["synthetic_motivic_infinite2"] == 1.0


def test_motivic_ext():
    assert bench_motivic_ext()["synthetic_motivic_ext"] == 1.0


def test_motivic_norm():
    assert bench_motivic_norm()["synthetic_motivic_norm"] == 1.0


def test_motivic_ramified():
    assert bench_motivic_ramified()["synthetic_motivic_ramified"] == 1.0


def test_motivic_euler():
    assert bench_motivic_euler()["synthetic_motivic_euler"] == 1.0
