from quant_fund.models.excellent_ring import (
    bench_excellent_ring,
)
from quant_fund.models.going_up import bench_going_up
from quant_fund.models.integral_closure2 import (
    bench_integral_closure2,
)
from quant_fund.models.lying_over import bench_lying_over
from quant_fund.models.weil_divisor2 import (
    bench_weil_divisor2,
)
from quant_fund.models.zariski_main import (
    bench_zariski_main,
)


def test_excellent_ring():
    assert bench_excellent_ring()["synthetic_excellent_ring"] == 1.0


def test_zariski_main():
    assert bench_zariski_main()["synthetic_zariski_main"] == 1.0


def test_going_up():
    assert bench_going_up()["synthetic_going_up"] == 1.0


def test_lying_over():
    assert bench_lying_over()["synthetic_lying_over"] == 1.0


def test_integral_closure2():
    assert bench_integral_closure2()["synthetic_integral_closure2"] == 1.0


def test_weil_divisor2():
    assert bench_weil_divisor2()["synthetic_weil_divisor2"] == 1.0
