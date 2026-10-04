"""Wave-1025 social-science canon tests."""

from __future__ import annotations

from quant_fund.models.behavioral_econ import bench_behavioral_econ
from quant_fund.models.cognitive_science import bench_cognitive_science
from quant_fund.models.game_theory2 import bench_game_theory2
from quant_fund.models.linguistics import bench_linguistics
from quant_fund.models.political_science import bench_political_science
from quant_fund.models.sociology_net import bench_sociology_net


def test_game_theory2():
    assert bench_game_theory2()["synthetic_game_theory2"] == 1.0


def test_behavioral_econ():
    assert bench_behavioral_econ()["synthetic_behavioral_econ"] == 1.0


def test_political_science():
    assert bench_political_science()["synthetic_political_science"] == 1.0


def test_sociology_net():
    assert bench_sociology_net()["synthetic_sociology_net"] == 1.0


def test_cognitive_science():
    assert bench_cognitive_science()["synthetic_cognitive_science"] == 1.0


def test_linguistics():
    assert bench_linguistics()["synthetic_linguistics"] == 1.0
