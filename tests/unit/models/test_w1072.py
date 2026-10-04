"""Wave-1072 film studies canon tests."""

from __future__ import annotations

from quant_fund.models.cinema_studies import bench_cinema_studies
from quant_fund.models.documentary_studies import bench_documentary_studies
from quant_fund.models.film_history import bench_film_history
from quant_fund.models.film_studies import bench_film_studies
from quant_fund.models.film_theory import bench_film_theory
from quant_fund.models.screenwriting import bench_screenwriting


def test_film_studies():
    assert bench_film_studies()["synthetic_film_studies"] == 1.0


def test_cinema_studies():
    assert bench_cinema_studies()["synthetic_cinema_studies"] == 1.0


def test_film_theory():
    assert bench_film_theory()["synthetic_film_theory"] == 1.0


def test_film_history():
    assert bench_film_history()["synthetic_film_history"] == 1.0


def test_documentary_studies():
    assert bench_documentary_studies()["synthetic_documentary_studies"] == 1.0


def test_screenwriting():
    assert bench_screenwriting()["synthetic_screenwriting"] == 1.0
