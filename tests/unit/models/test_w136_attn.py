"""Wave-136 attention-efficiency canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval
from quant_fund.models.linear_attn import bench_linear_attn
from quant_fund.models.linformer_attn import bench_linformer_attn
from quant_fund.models.nystrom_attn import bench_nystrom_attn
from quant_fund.models.performer_attn import bench_performer_attn
from quant_fund.models.sinkhorn_attn import bench_sinkhorn_attn
from quant_fund.models.sliding_attn import bench_sliding_attn


class TestFixture:
    def test_synth(self) -> None:
        x, y = synth_retrieval(8, 6, 4, np.random.default_rng(0))
        assert x.shape == (8, 13, 76) and y.shape == (8,)

    def test_cost(self) -> None:
        assert attn_dot_cost(100, "full", 1) == 1.0
        assert attn_dot_cost(100, "linformer", 10) < 1.0


class TestLinformer:
    def test_bench(self) -> None:
        out = bench_linformer_attn(seed=3, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_linformer_acc"] <= 1


class TestPerformer:
    def test_bench(self) -> None:
        out = bench_performer_attn(seed=5, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_performer_acc"] <= 1


class TestLinear:
    def test_bench(self) -> None:
        out = bench_linear_attn(seed=7, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_linear_acc"] <= 1


class TestSliding:
    def test_bench(self) -> None:
        out = bench_sliding_attn(seed=9, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_sliding_acc"] <= 1


class TestSinkhorn:
    def test_bench(self) -> None:
        out = bench_sinkhorn_attn(seed=11, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_sinkhorn_acc"] <= 1


class TestNystrom:
    def test_bench(self) -> None:
        out = bench_nystrom_attn(seed=13, n_train=100, n_test=40, m_pairs=16, iters=50)
        assert 0 <= out["synthetic_nystrom_acc"] <= 1
