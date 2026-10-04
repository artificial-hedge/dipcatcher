"""Wave-924 Bayesian-nonparametrics-2 canon tests."""

from __future__ import annotations

from quant_fund.models.beta_bernoulli import bench_beta_bernoulli
from quant_fund.models.crp_table import bench_crp_table
from quant_fund.models.dp_mm import bench_dp_mm
from quant_fund.models.gem_distribution import bench_gem_distribution
from quant_fund.models.gibbs_type import bench_gibbs_type
from quant_fund.models.neutral_process import bench_neutral_process


def test_gem_distribution():
    assert bench_gem_distribution()["synthetic_gem_distribution"] == 1.0


def test_dp_mm():
    assert bench_dp_mm()["synthetic_dp_mm"] == 1.0


def test_crp_table():
    assert bench_crp_table()["synthetic_crp_table"] == 1.0


def test_beta_bernoulli():
    assert bench_beta_bernoulli()["synthetic_beta_bernoulli"] == 1.0


def test_neutral_process():
    assert bench_neutral_process()["synthetic_neutral_process"] == 1.0


def test_gibbs_type():
    assert bench_gibbs_type()["synthetic_gibbs_type"] == 1.0
