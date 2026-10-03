from quant_fund.models.ek_subfactor import (
    bench_ek_subfactor,
)
from quant_fund.models.gyro_cat import bench_gyro_cat
from quant_fund.models.haagerup_sub import (
    bench_haagerup_sub,
)
from quant_fund.models.sovereign_cat import (
    bench_sovereign_cat,
)
from quant_fund.models.sylleptic import bench_sylleptic
from quant_fund.models.yang_lee_cat import (
    bench_yang_lee_cat,
)


def test_sylleptic():
    assert bench_sylleptic()["synthetic_sylleptic"] == 1.0


def test_haagerup_sub():
    assert bench_haagerup_sub()["synthetic_haagerup_sub"] == 1.0


def test_ek_subfactor():
    assert bench_ek_subfactor()["synthetic_ek_subfactor"] == 1.0


def test_gyro_cat():
    assert bench_gyro_cat()["synthetic_gyro_cat"] == 1.0


def test_yang_lee_cat():
    assert bench_yang_lee_cat()["synthetic_yang_lee_cat"] == 1.0


def test_sovereign_cat():
    assert bench_sovereign_cat()["synthetic_sovereign_cat"] == 1.0
