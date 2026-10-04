"""Wave-1210 psychiatry canon tests."""

from __future__ import annotations

from quant_fund.models.anxiety_disorders import bench_anxiety_disorders
from quant_fund.models.forensic_psychiatry import bench_forensic_psychiatry
from quant_fund.models.geriatric_psychiatry import bench_geriatric_psychiatry
from quant_fund.models.mood_disorders import bench_mood_disorders
from quant_fund.models.personality_disorders import bench_personality_disorders
from quant_fund.models.psychotic_disorders import bench_psychotic_disorders


def test_forensic_psychiatry():
    assert bench_forensic_psychiatry()["synthetic_forensic_psychiatry"] == 1.0


def test_geriatric_psychiatry():
    assert bench_geriatric_psychiatry()["synthetic_geriatric_psychiatry"] == 1.0


def test_mood_disorders():
    assert bench_mood_disorders()["synthetic_mood_disorders"] == 1.0


def test_psychotic_disorders():
    assert bench_psychotic_disorders()["synthetic_psychotic_disorders"] == 1.0


def test_personality_disorders():
    assert bench_personality_disorders()["synthetic_personality_disorders"] == 1.0


def test_anxiety_disorders():
    assert bench_anxiety_disorders()["synthetic_anxiety_disorders"] == 1.0
