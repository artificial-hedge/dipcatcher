"""Wave-940 variational-inequality canon tests."""

from __future__ import annotations

from quant_fund.models.forward_reflected import bench_forward_reflected
from quant_fund.models.korpelevich_eg import bench_korpelevich_eg
from quant_fund.models.popov_alg import bench_popov_alg
from quant_fund.models.reflected_golden import bench_reflected_golden
from quant_fund.models.subgradient_extragradient import bench_subgradient_extragradient
from quant_fund.models.tseng_fb import bench_tseng_fb


def test_subgradient_extragradient():
    assert bench_subgradient_extragradient()["synthetic_subgradient_extragradient"] == 1.0


def test_korpelevich_eg():
    assert bench_korpelevich_eg()["synthetic_korpelevich_eg"] == 1.0


def test_popov_alg():
    assert bench_popov_alg()["synthetic_popov_alg"] == 1.0


def test_tseng_fb():
    assert bench_tseng_fb()["synthetic_tseng_fb"] == 1.0


def test_forward_reflected():
    assert bench_forward_reflected()["synthetic_forward_reflected"] == 1.0


def test_reflected_golden():
    assert bench_reflected_golden()["synthetic_reflected_golden"] == 1.0
