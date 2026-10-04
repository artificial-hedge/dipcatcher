"""Wave-1124 sociology-4 canon tests."""

from __future__ import annotations

from quant_fund.models.sociology_of_aging import bench_sociology_of_aging
from quant_fund.models.sociology_of_emotions import bench_sociology_of_emotions
from quant_fund.models.sociology_of_food import bench_sociology_of_food
from quant_fund.models.sociology_of_media import bench_sociology_of_media
from quant_fund.models.sociology_of_sport import bench_sociology_of_sport
from quant_fund.models.sociology_of_work import bench_sociology_of_work


def test_sociology_of_work():
    assert bench_sociology_of_work()["synthetic_sociology_of_work"] == 1.0


def test_sociology_of_emotions():
    assert bench_sociology_of_emotions()["synthetic_sociology_of_emotions"] == 1.0


def test_sociology_of_food():
    assert bench_sociology_of_food()["synthetic_sociology_of_food"] == 1.0


def test_sociology_of_media():
    assert bench_sociology_of_media()["synthetic_sociology_of_media"] == 1.0


def test_sociology_of_sport():
    assert bench_sociology_of_sport()["synthetic_sociology_of_sport"] == 1.0


def test_sociology_of_aging():
    assert bench_sociology_of_aging()["synthetic_sociology_of_aging"] == 1.0
