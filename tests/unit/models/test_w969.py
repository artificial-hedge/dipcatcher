"""Wave-969 operator K-theory canon tests."""

from __future__ import annotations

from quant_fund.models.bott_periodicity_k import bench_bott_periodicity_k
from quant_fund.models.elliott_invariant import bench_elliott_invariant
from quant_fund.models.k0_algebra import bench_k0_algebra
from quant_fund.models.k1_algebra import bench_k1_algebra
from quant_fund.models.pimsner_voicul import bench_pimsner_voicul
from quant_fund.models.six_term_exact import bench_six_term_exact


def test_k0_algebra():
    assert bench_k0_algebra()["synthetic_k0_algebra"] == 1.0


def test_k1_algebra():
    assert bench_k1_algebra()["synthetic_k1_algebra"] == 1.0


def test_bott_periodicity_k():
    assert bench_bott_periodicity_k()["synthetic_bott_periodicity_k"] == 1.0


def test_six_term_exact():
    assert bench_six_term_exact()["synthetic_six_term_exact"] == 1.0


def test_pimsner_voicul():
    assert bench_pimsner_voicul()["synthetic_pimsner_voicul"] == 1.0


def test_elliott_invariant():
    assert bench_elliott_invariant()["synthetic_elliott_invariant"] == 1.0
