"""Wave-923 Bayesian-nonparametrics canon tests."""

from __future__ import annotations

from quant_fund.models.chinese_restaurant import bench_chinese_restaurant
from quant_fund.models.dirichlet_process import bench_dirichlet_process
from quant_fund.models.hierarchical_dp import bench_hierarchical_dp
from quant_fund.models.indian_buffet import bench_indian_buffet
from quant_fund.models.pitman_yor import bench_pitman_yor
from quant_fund.models.stick_breaking import bench_stick_breaking


def test_dirichlet_process():
    assert bench_dirichlet_process()["synthetic_dirichlet_process"] == 1.0


def test_stick_breaking():
    assert bench_stick_breaking()["synthetic_stick_breaking"] == 1.0


def test_pitman_yor():
    assert bench_pitman_yor()["synthetic_pitman_yor"] == 1.0


def test_indian_buffet():
    assert bench_indian_buffet()["synthetic_indian_buffet"] == 1.0


def test_chinese_restaurant():
    assert bench_chinese_restaurant()["synthetic_chinese_restaurant"] == 1.0


def test_hierarchical_dp():
    assert bench_hierarchical_dp()["synthetic_hierarchical_dp"] == 1.0
