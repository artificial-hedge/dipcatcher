"""Wave-954 matrix-approximation canon tests."""

from __future__ import annotations

from quant_fund.models.condition_number import bench_condition_number
from quant_fund.models.low_rank_approx import bench_low_rank_approx
from quant_fund.models.matrix_truncate import bench_matrix_truncate
from quant_fund.models.nuclear_norm import bench_nuclear_norm
from quant_fund.models.rank_estimate import bench_rank_estimate
from quant_fund.models.spectral_threshold import bench_spectral_threshold


def test_low_rank_approx():
    assert bench_low_rank_approx()["synthetic_low_rank_approx"] == 1.0


def test_nuclear_norm():
    assert bench_nuclear_norm()["synthetic_nuclear_norm"] == 1.0


def test_spectral_threshold():
    assert bench_spectral_threshold()["synthetic_spectral_threshold"] == 1.0


def test_matrix_truncate():
    assert bench_matrix_truncate()["synthetic_matrix_truncate"] == 1.0


def test_rank_estimate():
    assert bench_rank_estimate()["synthetic_rank_estimate"] == 1.0


def test_condition_number():
    assert bench_condition_number()["synthetic_condition_number"] == 1.0
