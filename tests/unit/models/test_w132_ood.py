"""Wave-132 OOD-detection canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.energy_ood import bench_energy_ood
from quant_fund.models.gradient_norm_ood import bench_gradient_norm_ood
from quant_fund.models.knn_ood import bench_knn_ood
from quant_fund.models.mahalanobis_ood import bench_mahalanobis_ood, synth_id_ood
from quant_fund.models.max_softmax_ood import bench_max_softmax_ood
from quant_fund.models.vim_ood import bench_vim_ood


class TestMahalanobis:
    def test_synth(self) -> None:
        (x, y), xo = synth_id_ood(40, 12, 8, np.random.default_rng(0))
        assert x.shape == (40, 8) and xo.shape == (12, 8)

    def test_bench(self) -> None:
        out = bench_mahalanobis_ood(seed=3, n_id=200, n_ood=50)
        assert out["synthetic_maha_auc"] > 0.5


class TestMaxSoftmax:
    def test_bench(self) -> None:
        out = bench_max_softmax_ood(seed=5, n_id=200, n_ood=50, iters=200)
        assert out["synthetic_msp_auc"] > 0.5


class TestGradientNorm:
    def test_bench(self) -> None:
        out = bench_gradient_norm_ood(seed=7, n_id=200, n_ood=50, iters=200)
        assert 0 <= out["synthetic_gradnorm_auc"] <= 1


class TestEnergyOod:
    def test_bench(self) -> None:
        out = bench_energy_ood(seed=11, n_id=200, n_ood=50, iters=200)
        assert out["synthetic_energyood_auc"] > 0.5


class TestKnn:
    def test_bench(self) -> None:
        out = bench_knn_ood(seed=13, n_id=200, n_ood=50, iters=200, k_nn=3)
        assert out["synthetic_knnood_auc"] > 0.5


class TestVim:
    def test_bench(self) -> None:
        out = bench_vim_ood(seed=17, n_id=200, n_ood=50, iters=200)
        assert 0 <= out["synthetic_vim_auc"] <= 1
