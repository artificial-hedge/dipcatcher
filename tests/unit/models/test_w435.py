from quant_fund.models.derived_fiber import bench_derived_fiber
from quant_fund.models.derived_scheme import bench_derived_scheme
from quant_fund.models.quasi_coherent import bench_quasi_coherent
from quant_fund.models.shifted_symplectic import (
    bench_shifted_symplectic,
)
from quant_fund.models.spectral_scheme import bench_spectral_scheme
from quant_fund.models.virtual_class import bench_virtual_class


def test_derived_scheme():
    assert bench_derived_scheme()["synthetic_derived_scheme"] == 1.0


def test_quasi_coherent():
    assert bench_quasi_coherent()["synthetic_quasi_coherent"] == 1.0


def test_derived_fiber():
    assert bench_derived_fiber()["synthetic_derived_fiber"] == 1.0


def test_spectral_scheme():
    assert bench_spectral_scheme()["synthetic_spectral_scheme"] == 1.0


def test_virtual_class():
    assert bench_virtual_class()["synthetic_virtual_class"] == 1.0


def test_shifted_symplectic():
    assert bench_shifted_symplectic()["synthetic_shifted_symplectic"] == 1.0
