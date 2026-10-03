from quant_fund.models.finite_spectra import bench_finite_spectra
from quant_fund.models.moore_spec import bench_moore_spec
from quant_fund.models.peterson_stein import bench_peterson_stein
from quant_fund.models.primary_op import bench_primary_op
from quant_fund.models.secondary_op import bench_secondary_op
from quant_fund.models.steenrod_sq import bench_steenrod_sq


def test_primary_op():
    assert bench_primary_op()["synthetic_primary_op"] == 1.0


def test_secondary_op():
    assert bench_secondary_op()["synthetic_secondary_op"] == 1.0


def test_steenrod_sq():
    assert bench_steenrod_sq()["synthetic_steenrod_sq"] == 1.0


def test_peterson_stein():
    assert bench_peterson_stein()["synthetic_peterson_stein"] == 1.0


def test_moore_spec():
    assert bench_moore_spec()["synthetic_moore_spec"] == 1.0


def test_finite_spectra():
    assert bench_finite_spectra()["synthetic_finite_spectra"] == 1.0
