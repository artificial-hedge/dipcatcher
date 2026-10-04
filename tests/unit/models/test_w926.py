"""Wave-926 information-geometry-2 canon tests."""

from __future__ import annotations

from quant_fund.models.alpha_divergence import bench_alpha_divergence
from quant_fund.models.amari_connection import bench_amari_connection
from quant_fund.models.csiszar_div import bench_csiszar_div
from quant_fund.models.dual_connection import bench_dual_connection
from quant_fund.models.f_divergence import bench_f_divergence
from quant_fund.models.tsallis_entropy import bench_tsallis_entropy


def test_f_divergence():
    assert bench_f_divergence()["synthetic_f_divergence"] == 1.0


def test_alpha_divergence():
    assert bench_alpha_divergence()["synthetic_alpha_divergence"] == 1.0


def test_csiszar_div():
    assert bench_csiszar_div()["synthetic_csiszar_div"] == 1.0


def test_amari_connection():
    assert bench_amari_connection()["synthetic_amari_connection"] == 1.0


def test_dual_connection():
    assert bench_dual_connection()["synthetic_dual_connection"] == 1.0


def test_tsallis_entropy():
    assert bench_tsallis_entropy()["synthetic_tsallis_entropy"] == 1.0
