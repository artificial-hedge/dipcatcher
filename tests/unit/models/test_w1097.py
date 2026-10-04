"""Wave-1097 philosophy-3 canon tests."""

from __future__ import annotations

from quant_fund.models.eastern_philosophy import bench_eastern_philosophy
from quant_fund.models.moral_philosophy import bench_moral_philosophy
from quant_fund.models.philosophy_of_language import bench_philosophy_of_language
from quant_fund.models.philosophy_of_law import bench_philosophy_of_law
from quant_fund.models.philosophy_of_mind import bench_philosophy_of_mind
from quant_fund.models.political_philosophy import bench_political_philosophy


def test_moral_philosophy():
    assert bench_moral_philosophy()["synthetic_moral_philosophy"] == 1.0


def test_political_philosophy():
    assert bench_political_philosophy()["synthetic_political_philosophy"] == 1.0


def test_philosophy_of_mind():
    assert bench_philosophy_of_mind()["synthetic_philosophy_of_mind"] == 1.0


def test_philosophy_of_language():
    assert bench_philosophy_of_language()["synthetic_philosophy_of_language"] == 1.0


def test_philosophy_of_law():
    assert bench_philosophy_of_law()["synthetic_philosophy_of_law"] == 1.0


def test_eastern_philosophy():
    assert bench_eastern_philosophy()["synthetic_eastern_philosophy"] == 1.0
