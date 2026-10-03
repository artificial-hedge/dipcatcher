"""Wave-198 scheduling canon tests."""

from __future__ import annotations

from quant_fund.models.johnson_flowshop import bench_johnson_flowshop
from quant_fund.models.knapsack_dp import bench_knapsack_dp
from quant_fund.models.lpt_schedule import bench_lpt_schedule
from quant_fund.models.neh_heuristic import bench_neh_heuristic
from quant_fund.models.spt_weighted import bench_spt_weighted
from quant_fund.models.tsp_branchbound import bench_tsp_branchbound


class TestJohnson:
    def test_bench(self) -> None:
        out = bench_johnson_flowshop()
        assert out["synthetic_johnson_gap"] < 1e-9
        assert out["synthetic_johnson_ms"] <= out["synthetic_random_ms"]


class TestNEH:
    def test_bench(self) -> None:
        out = bench_neh_heuristic()
        assert out["synthetic_neh_ms"] <= out["synthetic_neh_random_mean"]


class TestLPT:
    def test_bench(self) -> None:
        out = bench_lpt_schedule()
        assert out["synthetic_lpt_lb_ratio"] < 4.0 / 3.0 + 0.01


class TestKnapsack:
    def test_bench(self) -> None:
        out = bench_knapsack_dp()
        assert out["synthetic_ks_opt"] >= out["synthetic_ks_greedy"]


class TestTSP:
    def test_bench(self) -> None:
        out = bench_tsp_branchbound()
        assert out["synthetic_tsp_heur"] >= out["synthetic_tsp_opt"] - 1e-9


class TestWSPT:
    def test_bench(self) -> None:
        out = bench_spt_weighted()
        assert out["synthetic_wspt_cost"] <= out["synthetic_fifo_cost"]
