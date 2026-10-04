"""Wave-1207 psychotherapy canon tests."""

from __future__ import annotations

from quant_fund.models.art_therapy import bench_art_therapy
from quant_fund.models.behavioral_therapy_cognitive import bench_behavioral_therapy_cognitive
from quant_fund.models.music_therapy import bench_music_therapy
from quant_fund.models.play_therapy import bench_play_therapy
from quant_fund.models.psychoanalysis_studies import bench_psychoanalysis_studies
from quant_fund.models.psychotherapy_studies import bench_psychotherapy_studies


def test_psychoanalysis_studies():
    assert bench_psychoanalysis_studies()["synthetic_psychoanalysis_studies"] == 1.0


def test_psychotherapy_studies():
    assert bench_psychotherapy_studies()["synthetic_psychotherapy_studies"] == 1.0


def test_behavioral_therapy_cognitive():
    assert bench_behavioral_therapy_cognitive()["synthetic_behavioral_therapy_cognitive"] == 1.0


def test_art_therapy():
    assert bench_art_therapy()["synthetic_art_therapy"] == 1.0


def test_music_therapy():
    assert bench_music_therapy()["synthetic_music_therapy"] == 1.0


def test_play_therapy():
    assert bench_play_therapy()["synthetic_play_therapy"] == 1.0
