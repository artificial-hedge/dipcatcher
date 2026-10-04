"""Wave-1167 performing-arts canon tests."""

from __future__ import annotations

from quant_fund.models.art_history_2 import bench_art_history_2
from quant_fund.models.dance_2 import bench_dance_2
from quant_fund.models.film_studies_3 import bench_film_studies_3
from quant_fund.models.music_2 import bench_music_2
from quant_fund.models.performance_studies_2 import bench_performance_studies_2
from quant_fund.models.theater_2 import bench_theater_2


def test_music_2():
    assert bench_music_2()["synthetic_music_2"] == 1.0


def test_theater_2():
    assert bench_theater_2()["synthetic_theater_2"] == 1.0


def test_dance_2():
    assert bench_dance_2()["synthetic_dance_2"] == 1.0


def test_film_studies_3():
    assert bench_film_studies_3()["synthetic_film_studies_3"] == 1.0


def test_art_history_2():
    assert bench_art_history_2()["synthetic_art_history_2"] == 1.0


def test_performance_studies_2():
    assert bench_performance_studies_2()["synthetic_performance_studies_2"] == 1.0
