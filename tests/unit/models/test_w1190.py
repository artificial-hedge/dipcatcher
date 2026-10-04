"""Wave-1190 visual-design canon tests."""

from __future__ import annotations

from quant_fund.models.graphic_design import bench_graphic_design
from quant_fund.models.motion_graphics import bench_motion_graphics
from quant_fund.models.photography_studies import bench_photography_studies
from quant_fund.models.print_media import bench_print_media
from quant_fund.models.typography_studies import bench_typography_studies
from quant_fund.models.web_design import bench_web_design


def test_graphic_design():
    assert bench_graphic_design()["synthetic_graphic_design"] == 1.0


def test_typography_studies():
    assert bench_typography_studies()["synthetic_typography_studies"] == 1.0


def test_photography_studies():
    assert bench_photography_studies()["synthetic_photography_studies"] == 1.0


def test_print_media():
    assert bench_print_media()["synthetic_print_media"] == 1.0


def test_web_design():
    assert bench_web_design()["synthetic_web_design"] == 1.0


def test_motion_graphics():
    assert bench_motion_graphics()["synthetic_motion_graphics"] == 1.0
