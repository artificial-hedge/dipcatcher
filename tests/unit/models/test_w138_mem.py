"""Wave-138 memory + world-model canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._mem_synth import synth_copy, synth_dynamics
from quant_fund.models.dnc_memory import bench_dnc_memory
from quant_fund.models.memorizing_transformer import bench_memorizing_transformer
from quant_fund.models.mpc_planning import bench_mpc_planning
from quant_fund.models.ntm_memory import bench_ntm_memory
from quant_fund.models.reformer_lsh import bench_reformer_lsh
from quant_fund.models.rssm_world import bench_rssm_world


class TestFixture:
    def test_copy(self) -> None:
        x, y = synth_copy(4, 5, np.random.default_rng(0))
        assert x.shape == (4, 11, 9) and y.shape == (4, 5)

    def test_dyn(self) -> None:
        x, u, y = synth_dynamics(4, 6, np.random.default_rng(0))
        assert x.shape == (4, 7, 2) and u.shape == (4, 6, 1) and y.shape == (4, 6, 2)


class TestLSH:
    def test_bench(self) -> None:
        out = bench_reformer_lsh(seed=3, n_train=80, n_test=40, m_pairs=16, iters=60)
        assert 0 <= out["synthetic_lsh_acc"] <= 1


class TestMem:
    def test_bench(self) -> None:
        out = bench_memorizing_transformer(seed=5, n_train=80, n_test=40, m_pairs=16, iters=60)
        assert 0 <= out["synthetic_memtr_acc"] <= 1


class TestNTM:
    def test_bench(self) -> None:
        out = bench_ntm_memory(seed=7, n_train=60, n_test=30, t=5, iters=40)
        assert 0 <= out["synthetic_ntm_copy_acc"] <= 1


class TestDNC:
    def test_bench(self) -> None:
        out = bench_dnc_memory(seed=9, n_train=60, n_test=30, t=5, iters=40)
        assert 0 <= out["synthetic_dnc_copy_acc"] <= 1


class TestRSSM:
    def test_bench(self) -> None:
        out = bench_rssm_world(seed=11, n_train=80, n_test=30, horizon=6, iters=40)
        assert out["synthetic_rssm_mse"] >= 0


class TestMPC:
    def test_bench(self) -> None:
        out = bench_mpc_planning(seed=13, n_train=80, n_ep=4, horizon=6, plan_steps=3, iters=40)
        assert out["synthetic_mpc_dist"] >= 0
