"""Wave-984 modulation-spaces canon tests."""

from __future__ import annotations

from quant_fund.models.ambiguity_fn import bench_ambiguity_fn
from quant_fund.models.feichtinger_alg import bench_feichtinger_alg
from quant_fund.models.gabor_frame import bench_gabor_frame
from quant_fund.models.modulation_space import bench_modulation_space
from quant_fund.models.short_time_ft import bench_short_time_ft
from quant_fund.models.wigner_dist import bench_wigner_dist


def test_modulation_space():
    assert bench_modulation_space()["synthetic_modulation_space"] == 1.0


def test_short_time_ft():
    assert bench_short_time_ft()["synthetic_short_time_ft"] == 1.0


def test_gabor_frame():
    assert bench_gabor_frame()["synthetic_gabor_frame"] == 1.0


def test_wigner_dist():
    assert bench_wigner_dist()["synthetic_wigner_dist"] == 1.0


def test_ambiguity_fn():
    assert bench_ambiguity_fn()["synthetic_ambiguity_fn"] == 1.0


def test_feichtinger_alg():
    assert bench_feichtinger_alg()["synthetic_feichtinger_alg"] == 1.0
