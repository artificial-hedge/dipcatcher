"""Wave-151 agentic canon tests."""

from __future__ import annotations

from quant_fund.models._agent_synth import bfs_solution, is_goal, transition
from quant_fund.models.judge_pairwise import bench_judge_pairwise
from quant_fund.models.multi_agent_pipeline import bench_multi_agent_pipeline
from quant_fund.models.plan_search import bench_plan_search
from quant_fund.models.react_loop import bench_react_loop
from quant_fund.models.reflexion_retry import bench_reflexion_retry
from quant_fund.models.toolformer_call import bench_toolformer_call


class TestFixture:
    def test_transition(self) -> None:
        assert 0 <= transition(50, 0) < 100

    def test_bfs(self) -> None:
        sol = bfs_solution(43, 12)
        assert sol is not None
        s = 43
        for a in sol:
            s = transition(s, a)
        assert is_goal(s)


class TestReact:
    def test_bench(self) -> None:
        out = bench_react_loop(seed=3, n_tasks=20)
        assert out["synthetic_react_gain"] >= 0


class TestTool:
    def test_bench(self) -> None:
        out = bench_toolformer_call(seed=5, n_tasks=60)
        assert 0 <= out["synthetic_tool_gated_solve"] <= 1


class TestPlan:
    def test_bench(self) -> None:
        out = bench_plan_search(seed=7, n_tasks=15)
        assert 0 <= out["synthetic_plan_beam_solve"] <= 1


class TestRefl:
    def test_bench(self) -> None:
        out = bench_reflexion_retry(seed=9, n_tasks=15)
        assert 0 <= out["synthetic_refl_retry_solve"] <= 1


class TestMap:
    def test_bench(self) -> None:
        out = bench_multi_agent_pipeline(seed=11, n_tasks=15)
        assert 0 <= out["synthetic_map_pipe_solve"] <= 1


class TestJudge:
    def test_bench(self) -> None:
        out = bench_judge_pairwise(seed=13, n_items=20)
        assert 0 <= out["synthetic_judge_agreement"] <= 1
