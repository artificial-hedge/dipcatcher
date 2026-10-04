"""Wave-929 information-geometry-4 canon tests."""

from __future__ import annotations

from quant_fund.models.bhat_distance import bench_bhat_distance
from quant_fund.models.chi_square_div import bench_chi_square_div
from quant_fund.models.d_total_var import bench_d_total_var
from quant_fund.models.hellinger_dist import bench_hellinger_dist
from quant_fund.models.jeffreys_div import bench_jeffreys_div
from quant_fund.models.mahalanobis_div import bench_mahalanobis_div


def test_mahalanobis_div():
    assert bench_mahalanobis_div()["synthetic_mahalanobis_div"] == 1.0


def test_bhat_distance():
    assert bench_bhat_distance()["synthetic_bhat_distance"] == 1.0


def test_hellinger_dist():
    assert bench_hellinger_dist()["synthetic_hellinger_dist"] == 1.0


def test_jeffreys_div():
    assert bench_jeffreys_div()["synthetic_jeffreys_div"] == 1.0


def test_d_total_var():
    assert bench_d_total_var()["synthetic_d_total_var"] == 1.0


def test_chi_square_div():
    assert bench_chi_square_div()["synthetic_chi_square_div"] == 1.0
