from quant_fund.models.absolute_motive import bench_absolute_motive
from quant_fund.models.etale_motive import bench_etale_motive
from quant_fund.models.motivic_heart import bench_motivic_heart
from quant_fund.models.motivic_realization import (
    bench_motivic_realization,
)
from quant_fund.models.motivic_thh import bench_motivic_thh
from quant_fund.models.relative_motive import bench_relative_motive


def test_motivic_thh():
    assert bench_motivic_thh()["synthetic_motivic_thh"] == 1.0


def test_motivic_realization():
    assert bench_motivic_realization()["synthetic_motivic_realization"] == 1.0


def test_etale_motive():
    assert bench_etale_motive()["synthetic_etale_motive"] == 1.0


def test_relative_motive():
    assert bench_relative_motive()["synthetic_relative_motive"] == 1.0


def test_absolute_motive():
    assert bench_absolute_motive()["synthetic_absolute_motive"] == 1.0


def test_motivic_heart():
    assert bench_motivic_heart()["synthetic_motivic_heart"] == 1.0
