"""Wave-130 offline-RL / sequence-decision module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cql_agent import bench_cql_agent
from quant_fund.models.cql_agent import synth_offline as cql_off
from quant_fund.models.decision_transformer import (
    bench_decision_transformer,
    synth_trajectories,
)
from quant_fund.models.gail_imitation import bench_gail_imitation, synth_expert
from quant_fund.models.iql_agent import bench_iql_agent
from quant_fund.models.sac_agent import bench_sac_agent
from quant_fund.models.trajectory_transformer import (
    bench_trajectory_transformer,
    synth_offline,
)


class TestDecisionTransformer:
    def test_trajs(self) -> None:
        s, a, r = synth_trajectories(8, 12, np.random.default_rng(0))
        assert s.shape == (8, 12, 4) and a.shape == (8, 12)

    def test_bench(self) -> None:
        out = bench_decision_transformer(seed=3, n_ep=60, iters=200)
        assert out["synthetic_dt_margin_vs_random"] > 0


class TestCql:
    def test_data(self) -> None:
        s, a, r, sn = cql_off(6, 10, np.random.default_rng(0), 0.5)
        assert s.shape[1] == 4 and a.shape == r.shape == sn.shape[0:1] * 0 + a.shape

    def test_bench(self) -> None:
        out = bench_cql_agent(seed=5, n_ep=60, iters=200)
        assert np.isfinite(out["synthetic_cql_reward"])


class TestIql:
    def test_bench(self) -> None:
        out = bench_iql_agent(seed=7, n_ep=60, iters=200)
        assert out["synthetic_iql_margin_vs_dataset"] > 0


class TestTrajectoryTransformer:
    def test_data(self) -> None:
        tr = synth_offline(4, 10, np.random.default_rng(0))
        assert len(tr) == 4 and len(tr[0]) == 10 and len(tr[0][0]) == 5

    def test_bench(self) -> None:
        out = bench_trajectory_transformer(seed=11, n_ep=60, iters=150)
        assert np.isfinite(out["synthetic_tt_reward"])


class TestSac:
    def test_bench(self) -> None:
        out = bench_sac_agent(seed=13, steps=800, batch=64)
        assert np.isfinite(out["synthetic_sac_reward"])


class TestGail:
    def test_expert(self) -> None:
        s, a = synth_expert(6, 10, np.random.default_rng(0))
        assert s.shape == (60, 4)

    def test_bench(self) -> None:
        out = bench_gail_imitation(seed=17, n_ep=40, iters=200)
        assert np.isfinite(out["synthetic_gail_reward"])
