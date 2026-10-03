"""Wave-215 stochastic-programming canon tests."""

from __future__ import annotations

from quant_fund.models.chance_scenario import bench_chance_scenario
from quant_fund.models.dro_wasserstein import bench_dro_wasserstein
from quant_fund.models.robust_budget import bench_robust_budget
from quant_fund.models.saa_consistency import bench_saa_consistency
from quant_fund.models.scenario_tree import bench_scenario_tree
from quant_fund.models.two_stage_lshaped import bench_two_stage_lshaped

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestLShaped:
    def test_bench(self) -> None:
        out = bench_two_stage_lshaped()
        _clean(out)
        assert out["synthetic_ls_gap"] < 0.05


class TestScenarioTree:
    def test_bench(self) -> None:
        out = bench_scenario_tree()
        _clean(out)
        assert out["synthetic_tree_evpi"] >= -1e-9
        assert out["synthetic_tree_vss"] >= -1e-9


class TestSAA:
    def test_bench(self) -> None:
        out = bench_saa_consistency()
        _clean(out)
        assert out["synthetic_saa_gap_400"] < out["synthetic_saa_gap_50"] + 0.01


class TestChance:
    def test_bench(self) -> None:
        out = bench_chance_scenario()
        _clean(out)
        assert out["synthetic_cc_viol_200"] < out["synthetic_cc_bound_200"] + 0.02
        assert out["synthetic_cc_bound_shrinks"] == 1.0


class TestDRO:
    def test_bench(self) -> None:
        out = bench_dro_wasserstein()
        _clean(out)
        assert out["synthetic_dro_gain"] >= -0.05


class TestRobustBudget:
    def test_bench(self) -> None:
        out = bench_robust_budget()
        _clean(out)
        assert (
            out["synthetic_rb_nominal"]
            >= out["synthetic_rb_gamma1"]
            >= out["synthetic_rb_worst"] - 1e-9
        )
