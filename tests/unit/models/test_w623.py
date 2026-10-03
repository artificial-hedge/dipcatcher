from quant_fund.models.a_infinity2 import bench_a_infinity2
from quant_fund.models.cyclic_operad import (
    bench_cyclic_operad,
)
from quant_fund.models.dendroidal2 import bench_dendroidal2
from quant_fund.models.e_infinity3 import bench_e_infinity3
from quant_fund.models.infty_operad2 import (
    bench_infty_operad2,
)
from quant_fund.models.operadic_nerve import (
    bench_operadic_nerve,
)


def test_dendroidal2():
    assert bench_dendroidal2()["synthetic_dendroidal2"] == 1.0


def test_operadic_nerve():
    assert bench_operadic_nerve()["synthetic_operadic_nerve"] == 1.0


def test_infty_operad2():
    assert bench_infty_operad2()["synthetic_infty_operad2"] == 1.0


def test_a_infinity2():
    assert bench_a_infinity2()["synthetic_a_infinity2"] == 1.0


def test_e_infinity3():
    assert bench_e_infinity3()["synthetic_e_infinity3"] == 1.0


def test_cyclic_operad():
    assert bench_cyclic_operad()["synthetic_cyclic_operad"] == 1.0
