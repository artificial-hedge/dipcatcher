"""Wave-133 neural-process / amortized-UQ canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.attentive_np import bench_attentive_np
from quant_fund.models.convnp import bench_convnp
from quant_fund.models.deep_kernel_gp import bench_deep_kernel_gp
from quant_fund.models.llaplace_gp import bench_llaplace_gp
from quant_fund.models.meta_uq import bench_meta_uq
from quant_fund.models.neural_process import bench_neural_process, synth_np_tasks


class TestNeuralProcess:
    def test_synth(self) -> None:
        xc, yc, xt, yt = synth_np_tasks(8, 5, 10, np.random.default_rng(0))
        assert xc.shape == (8, 5, 1) and yt.shape == (8, 10, 1)

    def test_bench(self) -> None:
        out = bench_neural_process(seed=3, n_train=60, n_test=10, iters=60)
        assert out["synthetic_np_mse"] >= 0


class TestAttentiveNP:
    def test_bench(self) -> None:
        out = bench_attentive_np(seed=5, n_train=60, n_test=10, iters=60)
        assert out["synthetic_anp_mse"] >= 0


class TestDeepKernelGP:
    def test_bench(self) -> None:
        out = bench_deep_kernel_gp(seed=7, n_train=40, n_test=8, iters=30)
        assert out["synthetic_dkgp_mse"] >= 0


class TestConvNP:
    def test_bench(self) -> None:
        out = bench_convnp(seed=11, n_train=60, n_test=10, iters=60)
        assert out["synthetic_convnp_mse"] >= 0


class TestMetaUQ:
    def test_bench(self) -> None:
        out = bench_meta_uq(seed=13, n_train=60, n_test=6, iters=40, inner_steps=1)
        assert out["synthetic_meta_mse"] >= 0


class TestLLaplaceGP:
    def test_bench(self) -> None:
        out = bench_llaplace_gp(seed=17, n_train=60, n_test=10, iters=60)
        assert 0 <= out["synthetic_llap_cov90"] <= 1
