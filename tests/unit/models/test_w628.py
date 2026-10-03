from quant_fund.models.dendroidal_seg import (
    bench_dendroidal_seg,
)
from quant_fund.models.higher_operad import (
    bench_higher_operad,
)
from quant_fund.models.moerdijk_weiss import (
    bench_moerdijk_weiss,
)
from quant_fund.models.operad_cat2 import (
    bench_operad_cat2,
)
from quant_fund.models.operad_infty3 import (
    bench_operad_infty3,
)
from quant_fund.models.operad_module import (
    bench_operad_module,
)


def test_moerdijk_weiss():
    assert bench_moerdijk_weiss()["synthetic_moerdijk_weiss"] == 1.0


def test_higher_operad():
    assert bench_higher_operad()["synthetic_higher_operad"] == 1.0


def test_operad_infty3():
    assert bench_operad_infty3()["synthetic_operad_infty3"] == 1.0


def test_operad_cat2():
    assert bench_operad_cat2()["synthetic_operad_cat2"] == 1.0


def test_dendroidal_seg():
    assert bench_dendroidal_seg()["synthetic_dendroidal_seg"] == 1.0


def test_operad_module():
    assert bench_operad_module()["synthetic_operad_module"] == 1.0
