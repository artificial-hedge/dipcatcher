"""Wave-1059 history canon tests."""

from __future__ import annotations

from quant_fund.models.ancient_history import bench_ancient_history
from quant_fund.models.economic_history import bench_economic_history
from quant_fund.models.historiography import bench_historiography
from quant_fund.models.intellectual_history import bench_intellectual_history
from quant_fund.models.medieval_history import bench_medieval_history
from quant_fund.models.modern_history import bench_modern_history


def test_historiography():
    assert bench_historiography()["synthetic_historiography"] == 1.0


def test_ancient_history():
    assert bench_ancient_history()["synthetic_ancient_history"] == 1.0


def test_medieval_history():
    assert bench_medieval_history()["synthetic_medieval_history"] == 1.0


def test_modern_history():
    assert bench_modern_history()["synthetic_modern_history"] == 1.0


def test_economic_history():
    assert bench_economic_history()["synthetic_economic_history"] == 1.0


def test_intellectual_history():
    assert bench_intellectual_history()["synthetic_intellectual_history"] == 1.0
