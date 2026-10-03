"""Wave-978 harmonic-analysis-2 canon tests."""

from __future__ import annotations

from quant_fund.models.bochner_riesz import bench_bochner_riesz
from quant_fund.models.hausdorff_young import bench_hausdorff_young
from quant_fund.models.lp_multiplier import bench_lp_multiplier
from quant_fund.models.oscillatory_int import bench_oscillatory_int
from quant_fund.models.restriction_est import bench_restriction_est
from quant_fund.models.strichartz_est import bench_strichartz_est


def test_hausdorff_young():
    assert bench_hausdorff_young()["synthetic_hausdorff_young"] == 1.0


def test_restriction_est():
    assert bench_restriction_est()["synthetic_restriction_est"] == 1.0


def test_bochner_riesz():
    assert bench_bochner_riesz()["synthetic_bochner_riesz"] == 1.0


def test_lp_multiplier():
    assert bench_lp_multiplier()["synthetic_lp_multiplier"] == 1.0


def test_oscillatory_int():
    assert bench_oscillatory_int()["synthetic_oscillatory_int"] == 1.0


def test_strichartz_est():
    assert bench_strichartz_est()["synthetic_strichartz_est"] == 1.0
