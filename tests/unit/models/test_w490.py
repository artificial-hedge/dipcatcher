from quant_fund.models.adelic_curve import bench_adelic_curve
from quant_fund.models.arakelov_deg import bench_arakelov_deg
from quant_fund.models.arith_rr import bench_arith_rr
from quant_fund.models.arithmetic_chow import bench_arithmetic_chow
from quant_fund.models.faltings_metric import bench_faltings_metric
from quant_fund.models.height_arakelov import bench_height_arakelov


def test_arakelov_deg():
    assert bench_arakelov_deg()["synthetic_arakelov_deg"] == 1.0


def test_adelic_curve():
    assert bench_adelic_curve()["synthetic_adelic_curve"] == 1.0


def test_height_arakelov():
    assert bench_height_arakelov()["synthetic_height_arakelov"] == 1.0


def test_faltings_metric():
    assert bench_faltings_metric()["synthetic_faltings_metric"] == 1.0


def test_arithmetic_chow():
    assert bench_arithmetic_chow()["synthetic_arithmetic_chow"] == 1.0


def test_arith_rr():
    assert bench_arith_rr()["synthetic_arith_rr"] == 1.0
