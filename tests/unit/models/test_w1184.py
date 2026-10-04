"""Wave-1184 film-production canon tests."""

from __future__ import annotations

from quant_fund.models.animation_studies import bench_animation_studies
from quant_fund.models.cinematography_studies import bench_cinematography_studies
from quant_fund.models.documentary_production import bench_documentary_production
from quant_fund.models.film_editing import bench_film_editing
from quant_fund.models.film_production import bench_film_production
from quant_fund.models.sound_design import bench_sound_design


def test_film_production():
    assert bench_film_production()["synthetic_film_production"] == 1.0


def test_cinematography_studies():
    assert bench_cinematography_studies()["synthetic_cinematography_studies"] == 1.0


def test_film_editing():
    assert bench_film_editing()["synthetic_film_editing"] == 1.0


def test_sound_design():
    assert bench_sound_design()["synthetic_sound_design"] == 1.0


def test_documentary_production():
    assert bench_documentary_production()["synthetic_documentary_production"] == 1.0


def test_animation_studies():
    assert bench_animation_studies()["synthetic_animation_studies"] == 1.0
