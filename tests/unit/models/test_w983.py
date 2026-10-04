"""Wave-983 Besov/Triebel-Lizorkin canon tests."""

from __future__ import annotations

from quant_fund.models.atoms_decomp import bench_atoms_decomp
from quant_fund.models.besov_embed import bench_besov_embed
from quant_fund.models.besov_space import bench_besov_space
from quant_fund.models.hardy_littlewood_max import bench_hardy_littlewood_max
from quant_fund.models.triebel_lizorkin import bench_triebel_lizorkin
from quant_fund.models.wavelet_char import bench_wavelet_char


def test_besov_space():
    assert bench_besov_space()["synthetic_besov_space"] == 1.0


def test_triebel_lizorkin():
    assert bench_triebel_lizorkin()["synthetic_triebel_lizorkin"] == 1.0


def test_atoms_decomp():
    assert bench_atoms_decomp()["synthetic_atoms_decomp"] == 1.0


def test_wavelet_char():
    assert bench_wavelet_char()["synthetic_wavelet_char"] == 1.0


def test_besov_embed():
    assert bench_besov_embed()["synthetic_besov_embed"] == 1.0


def test_hardy_littlewood_max():
    assert bench_hardy_littlewood_max()["synthetic_hardy_littlewood_max"] == 1.0
