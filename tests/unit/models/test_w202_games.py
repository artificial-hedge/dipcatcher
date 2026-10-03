"""Wave-202 game-theory canon tests."""

from __future__ import annotations

from quant_fund.models.mfg_flocking import bench_mfg_flocking
from quant_fund.models.mfg_lq import bench_mfg_lq
from quant_fund.models.nash_cournot import bench_nash_cournot
from quant_fund.models.potential_game import bench_potential_game
from quant_fund.models.stackelberg_game import bench_stackelberg_game
from quant_fund.models.stochastic_game_vi import bench_stochastic_game_vi

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestMFGLQ:
    def test_bench(self) -> None:
        out = bench_mfg_lq()
        _clean(out)
        assert out["synthetic_mfg_resid"] < 1e-6
        assert out["synthetic_mfg_gain_shift"] > 0.001


class TestFlocking:
    def test_bench(self) -> None:
        out = bench_mfg_flocking()
        _clean(out)
        assert out["synthetic_flock_align_ratio"] < 1.0


class TestCournot:
    def test_bench(self) -> None:
        out = bench_nash_cournot()
        _clean(out)
        assert out["synthetic_cournot_resid"] < 1e-6
        assert out["synthetic_cournot_q"] > out["synthetic_cournot_monopoly_q"]


class TestStackelberg:
    def test_bench(self) -> None:
        out = bench_stackelberg_game()
        _clean(out)
        assert out["synthetic_stack_foc_resid"] < 1e-2
        assert out["synthetic_stack_leader_gain"] > 0.0


class TestStochGame:
    def test_bench(self) -> None:
        out = bench_stochastic_game_vi()
        _clean(out)
        assert out["synthetic_sg_resid"] < 1e-8


class TestPotential:
    def test_bench(self) -> None:
        out = bench_potential_game()
        _clean(out)
        assert out["synthetic_pot_nash_resid"] < 1e-6
        assert out["synthetic_pot_phi"] < out["synthetic_pot_phi_rand"]
