"""Wave-1183 music canon tests."""

from __future__ import annotations

from quant_fund.models.composition_studies import bench_composition_studies
from quant_fund.models.ethnomusicology_2 import bench_ethnomusicology_2
from quant_fund.models.music_cognition_2 import bench_music_cognition_2
from quant_fund.models.music_theory_2 import bench_music_theory_2
from quant_fund.models.musicology_2 import bench_musicology_2
from quant_fund.models.organology_2 import bench_organology_2


def test_music_theory_2():
    assert bench_music_theory_2()["synthetic_music_theory_2"] == 1.0


def test_musicology_2():
    assert bench_musicology_2()["synthetic_musicology_2"] == 1.0


def test_ethnomusicology_2():
    assert bench_ethnomusicology_2()["synthetic_ethnomusicology_2"] == 1.0


def test_music_cognition_2():
    assert bench_music_cognition_2()["synthetic_music_cognition_2"] == 1.0


def test_organology_2():
    assert bench_organology_2()["synthetic_organology_2"] == 1.0


def test_composition_studies():
    assert bench_composition_studies()["synthetic_composition_studies"] == 1.0
