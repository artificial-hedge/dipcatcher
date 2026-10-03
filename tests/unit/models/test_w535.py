from quant_fund.models.elliptic_est import bench_elliptic_est
from quant_fund.models.fourier_io import bench_fourier_io
from quant_fund.models.propagation_sing import bench_propagation_sing
from quant_fund.models.pseudodiff_op import bench_pseudodiff_op
from quant_fund.models.symbol_calc import bench_symbol_calc
from quant_fund.models.wavefront_set import bench_wavefront_set


def test_wavefront_set():
    assert bench_wavefront_set()["synthetic_wavefront_set"] == 1.0


def test_pseudodiff_op():
    assert bench_pseudodiff_op()["synthetic_pseudodiff_op"] == 1.0


def test_fourier_io():
    assert bench_fourier_io()["synthetic_fourier_io"] == 1.0


def test_symbol_calc():
    assert bench_symbol_calc()["synthetic_symbol_calc"] == 1.0


def test_propagation_sing():
    assert bench_propagation_sing()["synthetic_propagation_sing"] == 1.0


def test_elliptic_est():
    assert bench_elliptic_est()["synthetic_elliptic_est"] == 1.0
