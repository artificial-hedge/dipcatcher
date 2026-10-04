"""Wave-1127 history-3 canon tests."""

from __future__ import annotations

from quant_fund.models.digital_history import bench_digital_history
from quant_fund.models.environmental_history import bench_environmental_history
from quant_fund.models.global_history import bench_global_history
from quant_fund.models.maritime_history import bench_maritime_history
from quant_fund.models.oral_history import bench_oral_history
from quant_fund.models.public_history import bench_public_history


def test_oral_history():
    assert bench_oral_history()["synthetic_oral_history"] == 1.0


def test_public_history():
    assert bench_public_history()["synthetic_public_history"] == 1.0


def test_digital_history():
    assert bench_digital_history()["synthetic_digital_history"] == 1.0


def test_environmental_history():
    assert bench_environmental_history()["synthetic_environmental_history"] == 1.0


def test_global_history():
    assert bench_global_history()["synthetic_global_history"] == 1.0


def test_maritime_history():
    assert bench_maritime_history()["synthetic_maritime_history"] == 1.0
