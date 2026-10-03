"""Wave-927 information-geometry-3 canon tests."""

from __future__ import annotations

from quant_fund.models.ebanch_diverge import bench_ebanch_diverge
from quant_fund.models.expectation_param import bench_expectation_param
from quant_fund.models.fisher_metric2 import bench_fisher_metric2
from quant_fund.models.potential_fn import bench_potential_fn
from quant_fund.models.renyi_div import bench_renyi_div
from quant_fund.models.shannon_gibbs import bench_shannon_gibbs


def test_fisher_metric2():
    assert bench_fisher_metric2()["synthetic_fisher_metric2"] == 1.0


def test_expectation_param():
    assert bench_expectation_param()["synthetic_expectation_param"] == 1.0


def test_potential_fn():
    assert bench_potential_fn()["synthetic_potential_fn"] == 1.0


def test_ebanch_diverge():
    assert bench_ebanch_diverge()["synthetic_ebanch_diverge"] == 1.0


def test_shannon_gibbs():
    assert bench_shannon_gibbs()["synthetic_shannon_gibbs"] == 1.0


def test_renyi_div():
    assert bench_renyi_div()["synthetic_renyi_div"] == 1.0
