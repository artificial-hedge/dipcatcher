"""Wave-1079 classics canon tests."""

from __future__ import annotations

from quant_fund.models.ancient_greek import bench_ancient_greek
from quant_fund.models.classical_archaeology import bench_classical_archaeology
from quant_fund.models.classical_studies import bench_classical_studies
from quant_fund.models.latin_language import bench_latin_language
from quant_fund.models.papyrology import bench_papyrology
from quant_fund.models.philology import bench_philology


def test_classical_studies():
    assert bench_classical_studies()["synthetic_classical_studies"] == 1.0


def test_latin_language():
    assert bench_latin_language()["synthetic_latin_language"] == 1.0


def test_ancient_greek():
    assert bench_ancient_greek()["synthetic_ancient_greek"] == 1.0


def test_classical_archaeology():
    assert bench_classical_archaeology()["synthetic_classical_archaeology"] == 1.0


def test_philology():
    assert bench_philology()["synthetic_philology"] == 1.0


def test_papyrology():
    assert bench_papyrology()["synthetic_papyrology"] == 1.0
