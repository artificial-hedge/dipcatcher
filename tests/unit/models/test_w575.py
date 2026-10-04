from quant_fund.models.airy_process import bench_airy_process
from quant_fund.models.beta_ensemble import bench_beta_ensemble
from quant_fund.models.circular_law import bench_circular_law
from quant_fund.models.dyson_brownian import (
    bench_dyson_brownian,
)
from quant_fund.models.sine_kernel import bench_sine_kernel
from quant_fund.models.tracy_widom import bench_tracy_widom


def test_circular_law():
    assert bench_circular_law()["synthetic_circular_law"] == 1.0


def test_dyson_brownian():
    assert bench_dyson_brownian()["synthetic_dyson_brownian"] == 1.0


def test_sine_kernel():
    assert bench_sine_kernel()["synthetic_sine_kernel"] == 1.0


def test_airy_process():
    assert bench_airy_process()["synthetic_airy_process"] == 1.0


def test_tracy_widom():
    assert bench_tracy_widom()["synthetic_tracy_widom"] == 1.0


def test_beta_ensemble():
    assert bench_beta_ensemble()["synthetic_beta_ensemble"] == 1.0
