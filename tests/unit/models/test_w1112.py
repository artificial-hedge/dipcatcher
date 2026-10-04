"""Wave-1112 biology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.biophysics import bench_biophysics
from quant_fund.models.comparative_anatomy import bench_comparative_anatomy
from quant_fund.models.developmental_biology import bench_developmental_biology
from quant_fund.models.ethology import bench_ethology
from quant_fund.models.evolutionary_biology import bench_evolutionary_biology
from quant_fund.models.neurobiology import bench_neurobiology


def test_biophysics():
    assert bench_biophysics()["synthetic_biophysics"] == 1.0


def test_evolutionary_biology():
    assert bench_evolutionary_biology()["synthetic_evolutionary_biology"] == 1.0


def test_developmental_biology():
    assert bench_developmental_biology()["synthetic_developmental_biology"] == 1.0


def test_neurobiology():
    assert bench_neurobiology()["synthetic_neurobiology"] == 1.0


def test_ethology():
    assert bench_ethology()["synthetic_ethology"] == 1.0


def test_comparative_anatomy():
    assert bench_comparative_anatomy()["synthetic_comparative_anatomy"] == 1.0
