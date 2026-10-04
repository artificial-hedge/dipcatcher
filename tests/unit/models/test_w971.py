"""Wave-971 noncommutative-geometry canon tests."""

from __future__ import annotations

from quant_fund.models.connes_metric import bench_connes_metric
from quant_fund.models.differential_form_nc import bench_differential_form_nc
from quant_fund.models.geodesic_nc import bench_geodesic_nc
from quant_fund.models.hochschild_cycle import bench_hochschild_cycle
from quant_fund.models.index_pairing import bench_index_pairing
from quant_fund.models.spectral_triple import bench_spectral_triple


def test_spectral_triple():
    assert bench_spectral_triple()["synthetic_spectral_triple"] == 1.0


def test_connes_metric():
    assert bench_connes_metric()["synthetic_connes_metric"] == 1.0


def test_index_pairing():
    assert bench_index_pairing()["synthetic_index_pairing"] == 1.0


def test_hochschild_cycle():
    assert bench_hochschild_cycle()["synthetic_hochschild_cycle"] == 1.0


def test_differential_form_nc():
    assert bench_differential_form_nc()["synthetic_differential_form_nc"] == 1.0


def test_geodesic_nc():
    assert bench_geodesic_nc()["synthetic_geodesic_nc"] == 1.0
