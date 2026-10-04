from quant_fund.models.bernstein_sato import bench_bernstein_sato
from quant_fund.models.du_val_sing import bench_du_val_sing
from quant_fund.models.log_canonical import bench_log_canonical
from quant_fund.models.milnor_fiber import bench_milnor_fiber
from quant_fund.models.multiplier_ideal import (
    bench_multiplier_ideal,
)
from quant_fund.models.rational_sing import bench_rational_sing


def test_du_val_sing():
    assert bench_du_val_sing()["synthetic_du_val_sing"] == 1.0


def test_rational_sing():
    assert bench_rational_sing()["synthetic_rational_sing"] == 1.0


def test_log_canonical():
    assert bench_log_canonical()["synthetic_log_canonical"] == 1.0


def test_multiplier_ideal():
    assert bench_multiplier_ideal()["synthetic_multiplier_ideal"] == 1.0


def test_bernstein_sato():
    assert bench_bernstein_sato()["synthetic_bernstein_sato"] == 1.0


def test_milnor_fiber():
    assert bench_milnor_fiber()["synthetic_milnor_fiber"] == 1.0
