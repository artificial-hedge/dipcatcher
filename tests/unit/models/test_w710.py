from quant_fund.models.floyd_farey import bench_floyd_farey
from quant_fund.models.higher_algebra8 import (
    bench_higher_algebra8,
)
from quant_fund.models.little_discs3 import (
    bench_little_discs3,
)
from quant_fund.models.operad_infty4 import (
    bench_operad_infty4,
)
from quant_fund.models.operad_swiss3 import (
    bench_operad_swiss3,
)
from quant_fund.models.operad_twisted import (
    bench_operad_twisted,
)


def test_higher_algebra8():
    assert bench_higher_algebra8()["synthetic_higher_algebra8"] == 1.0


def test_operad_infty4():
    assert bench_operad_infty4()["synthetic_operad_infty4"] == 1.0


def test_floyd_farey():
    assert bench_floyd_farey()["synthetic_floyd_farey"] == 1.0


def test_operad_swiss3():
    assert bench_operad_swiss3()["synthetic_operad_swiss3"] == 1.0


def test_little_discs3():
    assert bench_little_discs3()["synthetic_little_discs3"] == 1.0


def test_operad_twisted():
    assert bench_operad_twisted()["synthetic_operad_twisted"] == 1.0
