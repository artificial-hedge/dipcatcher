"""Wave-925 Bayesian-nonparametrics-3 canon tests."""

from __future__ import annotations

from quant_fund.models.bondesson_shot import bench_bondesson_shot
from quant_fund.models.exchangeable_pf import bench_exchangeable_pf
from quant_fund.models.kingman_paintbox import bench_kingman_paintbox
from quant_fund.models.nggp_process import bench_nggp_process
from quant_fund.models.normalized_rm import bench_normalized_rm
from quant_fund.models.sigma_stable import bench_sigma_stable


def test_exchangeable_pf():
    assert bench_exchangeable_pf()["synthetic_exchangeable_pf"] == 1.0


def test_normalized_rm():
    assert bench_normalized_rm()["synthetic_normalized_rm"] == 1.0


def test_sigma_stable():
    assert bench_sigma_stable()["synthetic_sigma_stable"] == 1.0


def test_nggp_process():
    assert bench_nggp_process()["synthetic_nggp_process"] == 1.0


def test_bondesson_shot():
    assert bench_bondesson_shot()["synthetic_bondesson_shot"] == 1.0


def test_kingman_paintbox():
    assert bench_kingman_paintbox()["synthetic_kingman_paintbox"] == 1.0
