"""Wave-941 nonsmooth-Newton canon tests."""

from __future__ import annotations

from quant_fund.models.augmented_lagr import bench_augmented_lagr
from quant_fund.models.ekeland_var import bench_ekeland_var
from quant_fund.models.limiting_subdiff import bench_limiting_subdiff
from quant_fund.models.monteiro_semismooth import bench_monteiro_semismooth
from quant_fund.models.proximal_subdiff import bench_proximal_subdiff
from quant_fund.models.semismooth_newton import bench_semismooth_newton


def test_limiting_subdiff():
    assert bench_limiting_subdiff()["synthetic_limiting_subdiff"] == 1.0


def test_proximal_subdiff():
    assert bench_proximal_subdiff()["synthetic_proximal_subdiff"] == 1.0


def test_ekeland_var():
    assert bench_ekeland_var()["synthetic_ekeland_var"] == 1.0


def test_monteiro_semismooth():
    assert bench_monteiro_semismooth()["synthetic_monteiro_semismooth"] == 1.0


def test_semismooth_newton():
    assert bench_semismooth_newton()["synthetic_semismooth_newton"] == 1.0


def test_augmented_lagr():
    assert bench_augmented_lagr()["synthetic_augmented_lagr"] == 1.0
