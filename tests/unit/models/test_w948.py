"""Wave-948 structured-matrix canon tests."""

from __future__ import annotations

from quant_fund.models.circulant_matrix import bench_circulant_matrix
from quant_fund.models.companion_matrix import bench_companion_matrix
from quant_fund.models.hankel_matrix import bench_hankel_matrix
from quant_fund.models.hessenberg_form import bench_hessenberg_form
from quant_fund.models.krylov_matrix import bench_krylov_matrix
from quant_fund.models.vandermonde_matrix import bench_vandermonde_matrix


def test_circulant_matrix():
    assert bench_circulant_matrix()["synthetic_circulant_matrix"] == 1.0


def test_companion_matrix():
    assert bench_companion_matrix()["synthetic_companion_matrix"] == 1.0


def test_vandermonde_matrix():
    assert bench_vandermonde_matrix()["synthetic_vandermonde_matrix"] == 1.0


def test_krylov_matrix():
    assert bench_krylov_matrix()["synthetic_krylov_matrix"] == 1.0


def test_hessenberg_form():
    assert bench_hessenberg_form()["synthetic_hessenberg_form"] == 1.0


def test_hankel_matrix():
    assert bench_hankel_matrix()["synthetic_hankel_matrix"] == 1.0
