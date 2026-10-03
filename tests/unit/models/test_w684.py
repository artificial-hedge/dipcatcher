from quant_fund.models.motivic_base2 import bench_motivic_base2
from quant_fund.models.motivic_frequency import (
    bench_motivic_frequency,
)
from quant_fund.models.motivic_infinite import (
    bench_motivic_infinite,
)
from quant_fund.models.motivic_suslin import bench_motivic_suslin
from quant_fund.models.motivic_weight2 import bench_motivic_weight2
from quant_fund.models.motivic_wit import bench_motivic_wit


def test_motivic_weight2():
    assert bench_motivic_weight2()["synthetic_motivic_weight2"] == 1.0


def test_motivic_infinite():
    assert bench_motivic_infinite()["synthetic_motivic_infinite"] == 1.0


def test_motivic_suslin():
    assert bench_motivic_suslin()["synthetic_motivic_suslin"] == 1.0


def test_motivic_frequency():
    assert bench_motivic_frequency()["synthetic_motivic_frequency"] == 1.0


def test_motivic_wit():
    assert bench_motivic_wit()["synthetic_motivic_wit"] == 1.0


def test_motivic_base2():
    assert bench_motivic_base2()["synthetic_motivic_base2"] == 1.0
