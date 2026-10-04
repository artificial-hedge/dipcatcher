"""Wave-911 computational-geometry-2 canon tests."""

from __future__ import annotations

from quant_fund.models.bezier_eval import bench_bezier_eval
from quant_fund.models.chan_hull import bench_chan_hull
from quant_fund.models.cohen_sutherland import bench_cohen_sutherland
from quant_fund.models.gift_wrap import bench_gift_wrap
from quant_fund.models.liang_barsky import bench_liang_barsky
from quant_fund.models.monotone_chain import bench_monotone_chain


def test_monotone_chain():
    assert bench_monotone_chain()["synthetic_monotone_chain"] == 1.0


def test_gift_wrap():
    assert bench_gift_wrap()["synthetic_gift_wrap"] == 1.0


def test_chan_hull():
    assert bench_chan_hull()["synthetic_chan_hull"] == 1.0


def test_liang_barsky():
    assert bench_liang_barsky()["synthetic_liang_barsky"] == 1.0


def test_cohen_sutherland():
    assert bench_cohen_sutherland()["synthetic_cohen_sutherland"] == 1.0


def test_bezier_eval():
    assert bench_bezier_eval()["synthetic_bezier_eval"] == 1.0
