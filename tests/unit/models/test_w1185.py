"""Wave-1185 game canon tests."""

from __future__ import annotations

from quant_fund.models.esports_studies import bench_esports_studies
from quant_fund.models.game_design import bench_game_design
from quant_fund.models.game_development import bench_game_development
from quant_fund.models.game_studies import bench_game_studies
from quant_fund.models.interactive_media import bench_interactive_media
from quant_fund.models.ludology import bench_ludology


def test_game_design():
    assert bench_game_design()["synthetic_game_design"] == 1.0


def test_esports_studies():
    assert bench_esports_studies()["synthetic_esports_studies"] == 1.0


def test_interactive_media():
    assert bench_interactive_media()["synthetic_interactive_media"] == 1.0


def test_game_studies():
    assert bench_game_studies()["synthetic_game_studies"] == 1.0


def test_ludology():
    assert bench_ludology()["synthetic_ludology"] == 1.0


def test_game_development():
    assert bench_game_development()["synthetic_game_development"] == 1.0
