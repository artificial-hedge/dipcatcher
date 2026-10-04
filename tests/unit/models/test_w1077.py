"""Wave-1077 visual arts canon tests."""

from __future__ import annotations

from quant_fund.models.art_conservation import bench_art_conservation
from quant_fund.models.art_history import bench_art_history
from quant_fund.models.painting_techniques import bench_painting_techniques
from quant_fund.models.printmaking import bench_printmaking
from quant_fund.models.sculpture_methods import bench_sculpture_methods
from quant_fund.models.visual_culture import bench_visual_culture


def test_painting_techniques():
    assert bench_painting_techniques()["synthetic_painting_techniques"] == 1.0


def test_sculpture_methods():
    assert bench_sculpture_methods()["synthetic_sculpture_methods"] == 1.0


def test_printmaking():
    assert bench_printmaking()["synthetic_printmaking"] == 1.0


def test_art_conservation():
    assert bench_art_conservation()["synthetic_art_conservation"] == 1.0


def test_art_history():
    assert bench_art_history()["synthetic_art_history"] == 1.0


def test_visual_culture():
    assert bench_visual_culture()["synthetic_visual_culture"] == 1.0
