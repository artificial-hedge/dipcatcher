"""Wave-1099 psychology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.abnormal_psychology import bench_abnormal_psychology
from quant_fund.models.forensic_psychology import bench_forensic_psychology
from quant_fund.models.health_psychology import bench_health_psychology
from quant_fund.models.neuropsychology import bench_neuropsychology
from quant_fund.models.organizational_psychology import bench_organizational_psychology
from quant_fund.models.personality_psychology import bench_personality_psychology


def test_personality_psychology():
    assert bench_personality_psychology()["synthetic_personality_psychology"] == 1.0


def test_abnormal_psychology():
    assert bench_abnormal_psychology()["synthetic_abnormal_psychology"] == 1.0


def test_health_psychology():
    assert bench_health_psychology()["synthetic_health_psychology"] == 1.0


def test_neuropsychology():
    assert bench_neuropsychology()["synthetic_neuropsychology"] == 1.0


def test_forensic_psychology():
    assert bench_forensic_psychology()["synthetic_forensic_psychology"] == 1.0


def test_organizational_psychology():
    assert bench_organizational_psychology()["synthetic_organizational_psychology"] == 1.0
