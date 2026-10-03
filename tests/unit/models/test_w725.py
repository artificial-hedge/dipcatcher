from quant_fund.models.arithmetic_arnold import (
    bench_arithmetic_arnold,
)
from quant_fund.models.bertolini_darmon import (
    bench_bertolini_darmon,
)
from quant_fund.models.darmon_point import bench_darmon_point
from quant_fund.models.howard_main import bench_howard_main
from quant_fund.models.p_group_iwasawa import (
    bench_p_group_iwasawa,
)
from quant_fund.models.shimura_period import (
    bench_shimura_period,
)


def test_p_group_iwasawa():
    assert bench_p_group_iwasawa()["synthetic_p_group_iwasawa"] == 1.0


def test_shimura_period():
    assert bench_shimura_period()["synthetic_shimura_period"] == 1.0


def test_arithmetic_arnold():
    assert bench_arithmetic_arnold()["synthetic_arithmetic_arnold"] == 1.0


def test_darmon_point():
    assert bench_darmon_point()["synthetic_darmon_point"] == 1.0


def test_bertolini_darmon():
    assert bench_bertolini_darmon()["synthetic_bertolini_darmon"] == 1.0


def test_howard_main():
    assert bench_howard_main()["synthetic_howard_main"] == 1.0
