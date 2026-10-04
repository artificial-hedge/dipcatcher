from quant_fund.models.derived_conn import bench_derived_conn
from quant_fund.models.derived_integral import (
    bench_derived_integral,
)
from quant_fund.models.derived_local import bench_derived_local
from quant_fund.models.derived_noether import (
    bench_derived_noether,
)
from quant_fund.models.derived_normal import bench_derived_normal
from quant_fund.models.derived_reduced import (
    bench_derived_reduced,
)


def test_derived_conn():
    assert bench_derived_conn()["synthetic_derived_conn"] == 1.0


def test_derived_local():
    assert bench_derived_local()["synthetic_derived_local"] == 1.0


def test_derived_reduced():
    assert bench_derived_reduced()["synthetic_derived_reduced"] == 1.0


def test_derived_integral():
    assert bench_derived_integral()["synthetic_derived_integral"] == 1.0


def test_derived_normal():
    assert bench_derived_normal()["synthetic_derived_normal"] == 1.0


def test_derived_noether():
    assert bench_derived_noether()["synthetic_derived_noether"] == 1.0
