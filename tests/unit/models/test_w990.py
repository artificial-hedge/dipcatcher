"""Wave-990 calculus-of-variations canon tests."""

from __future__ import annotations

from quant_fund.models.euler_lagrange import bench_euler_lagrange
from quant_fund.models.geodesic_var import bench_geodesic_var
from quant_fund.models.isoperimetric_var import bench_isoperimetric_var
from quant_fund.models.jacobi_eq import bench_jacobi_eq
from quant_fund.models.legendre_cond import bench_legendre_cond
from quant_fund.models.soap_film import bench_soap_film


def test_euler_lagrange():
    assert bench_euler_lagrange()["synthetic_euler_lagrange"] == 1.0


def test_legendre_cond():
    assert bench_legendre_cond()["synthetic_legendre_cond"] == 1.0


def test_jacobi_eq():
    assert bench_jacobi_eq()["synthetic_jacobi_eq"] == 1.0


def test_geodesic_var():
    assert bench_geodesic_var()["synthetic_geodesic_var"] == 1.0


def test_isoperimetric_var():
    assert bench_isoperimetric_var()["synthetic_isoperimetric_var"] == 1.0


def test_soap_film():
    assert bench_soap_film()["synthetic_soap_film"] == 1.0
