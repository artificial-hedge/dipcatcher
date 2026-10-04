"""Wave-1071 musicology canon tests."""

from __future__ import annotations

from quant_fund.models.ethnomusicology import bench_ethnomusicology
from quant_fund.models.music_cognition import bench_music_cognition
from quant_fund.models.music_history import bench_music_history
from quant_fund.models.music_theory import bench_music_theory
from quant_fund.models.musicology import bench_musicology
from quant_fund.models.organology import bench_organology


def test_musicology():
    assert bench_musicology()["synthetic_musicology"] == 1.0


def test_ethnomusicology():
    assert bench_ethnomusicology()["synthetic_ethnomusicology"] == 1.0


def test_music_theory():
    assert bench_music_theory()["synthetic_music_theory"] == 1.0


def test_music_cognition():
    assert bench_music_cognition()["synthetic_music_cognition"] == 1.0


def test_organology():
    assert bench_organology()["synthetic_organology"] == 1.0


def test_music_history():
    assert bench_music_history()["synthetic_music_history"] == 1.0
