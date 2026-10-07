"""Wave-184 training-dynamics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.catapult_phase import bench_catapult_phase
from quant_fund.models.edge_stability import bench_edge_stability
from quant_fund.models.hessian_eig import bench_hessian_eig
from quant_fund.models.mode_connectivity import bench_mode_connectivity
from quant_fund.models.neural_grok import bench_neural_grok
from quant_fund.models.ntk_kernel import bench_ntk_kernel


class TestHessian:
    def test_bench(self) -> None:
        out = bench_hessian_eig(seed=3)
        assert np.isfinite(out["synthetic_hess_top_trained"])


class TestNTK:
    def test_bench(self) -> None:
        out = bench_ntk_kernel(seed=5)
        assert 0.0 <= out["synthetic_ntk_acc"] <= 1.0


class TestEOS:
    def test_bench(self) -> None:
        out = bench_edge_stability(seed=7, iters=60)
        assert np.isfinite(out["synthetic_eos_mean_sharpness"])


class TestLMC:
    def test_bench(self) -> None:
        out = bench_mode_connectivity(seed=9)
        assert np.isfinite(out["synthetic_lmc_barrier"])


class TestCatapult:
    def test_bench(self) -> None:
        out = bench_catapult_phase(seed=11)
        assert np.isfinite(out["synthetic_catapult_spike"])


class TestGrok:
    def test_bench(self) -> None:
        out = bench_neural_grok(seed=13, iters=100)
        assert 0.0 <= out["synthetic_grok_test_acc"] <= 1.0


class TestTrainMLPRNG:
    def test_preserves_global_rng(self) -> None:
        import torch

        from quant_fund.models._td_synth import train_mlp

        X = np.zeros((8, 8))
        y = np.zeros(8)
        torch.manual_seed(123)
        before = torch.rand(3)
        torch.manual_seed(123)
        train_mlp(torch, X, y, iters=2, seed=5)
        assert torch.equal(before, torch.rand(3))
