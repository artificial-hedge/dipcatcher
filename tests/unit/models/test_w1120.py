"""Wave-1120 anthropology-4 canon tests."""

from __future__ import annotations

from quant_fund.models.applied_anthropology import bench_applied_anthropology
from quant_fund.models.digital_anthropology import bench_digital_anthropology
from quant_fund.models.environmental_anthropology import bench_environmental_anthropology
from quant_fund.models.forensic_anthropology import bench_forensic_anthropology
from quant_fund.models.psychological_anthropology import bench_psychological_anthropology
from quant_fund.models.visual_anthropology import bench_visual_anthropology


def test_visual_anthropology():
    assert bench_visual_anthropology()["synthetic_visual_anthropology"] == 1.0


def test_applied_anthropology():
    assert bench_applied_anthropology()["synthetic_applied_anthropology"] == 1.0


def test_forensic_anthropology():
    assert bench_forensic_anthropology()["synthetic_forensic_anthropology"] == 1.0


def test_digital_anthropology():
    assert bench_digital_anthropology()["synthetic_digital_anthropology"] == 1.0


def test_environmental_anthropology():
    assert bench_environmental_anthropology()["synthetic_environmental_anthropology"] == 1.0


def test_psychological_anthropology():
    assert bench_psychological_anthropology()["synthetic_psychological_anthropology"] == 1.0
