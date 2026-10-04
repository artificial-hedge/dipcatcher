from quant_fund.models.motivic_degree import (
    bench_motivic_degree,
)
from quant_fund.models.motivic_diagonal import (
    bench_motivic_diagonal,
)
from quant_fund.models.motivic_field import bench_motivic_field
from quant_fund.models.motivic_fundamental import (
    bench_motivic_fundamental,
)
from quant_fund.models.motivic_hochschild import (
    bench_motivic_hochschild,
)
from quant_fund.models.motivic_spark import bench_motivic_spark


def test_motivic_spark():
    assert bench_motivic_spark()["synthetic_motivic_spark"] == 1.0


def test_motivic_fundamental():
    assert bench_motivic_fundamental()["synthetic_motivic_fundamental"] == 1.0


def test_motivic_hochschild():
    assert bench_motivic_hochschild()["synthetic_motivic_hochschild"] == 1.0


def test_motivic_field():
    assert bench_motivic_field()["synthetic_motivic_field"] == 1.0


def test_motivic_degree():
    assert bench_motivic_degree()["synthetic_motivic_degree"] == 1.0


def test_motivic_diagonal():
    assert bench_motivic_diagonal()["synthetic_motivic_diagonal"] == 1.0
