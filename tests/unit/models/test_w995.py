"""Wave-995 integrable-systems canon tests."""

from __future__ import annotations

from quant_fund.models.calogero_moser import bench_calogero_moser
from quant_fund.models.kp_hierarchy import bench_kp_hierarchy
from quant_fund.models.nls_soliton import bench_nls_soliton
from quant_fund.models.painleve_eq import bench_painleve_eq
from quant_fund.models.sine_gordon import bench_sine_gordon
from quant_fund.models.toda_lattice import bench_toda_lattice


def test_sine_gordon():
    assert bench_sine_gordon()["synthetic_sine_gordon"] == 1.0


def test_nls_soliton():
    assert bench_nls_soliton()["synthetic_nls_soliton"] == 1.0


def test_toda_lattice():
    assert bench_toda_lattice()["synthetic_toda_lattice"] == 1.0


def test_calogero_moser():
    assert bench_calogero_moser()["synthetic_calogero_moser"] == 1.0


def test_kp_hierarchy():
    assert bench_kp_hierarchy()["synthetic_kp_hierarchy"] == 1.0


def test_painleve_eq():
    assert bench_painleve_eq()["synthetic_painleve_eq"] == 1.0
