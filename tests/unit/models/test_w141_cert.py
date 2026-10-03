"""Wave-141 certified-robustness canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._cert_synth import synth_cls_2d, synth_subset
from quant_fund.models.crown_bound import bench_crown_bound
from quant_fund.models.gumbel_topk import bench_gumbel_topk
from quant_fund.models.ibp_bounds import bench_ibp_bounds
from quant_fund.models.lipschitz_net import bench_lipschitz_net
from quant_fund.models.randomized_smoothing import bench_randomized_smoothing
from quant_fund.models.vector_neurons import bench_vector_neurons


class TestFixture:
    def test_cls(self) -> None:
        x, y = synth_cls_2d(30, np.random.default_rng(0))
        assert x.shape[1] == 4 and y.max() <= 1

    def test_subset(self) -> None:
        x, y, m = synth_subset(30, 8, 3, np.random.default_rng(0))
        assert x.shape == (30, 8) and m.sum() == 3


class TestRS:
    def test_bench(self) -> None:
        out = bench_randomized_smoothing(seed=3, n_train=80, n_test=30, n_samples=20, iters=40)
        assert 0 <= out["synthetic_smooth_acc"] <= 1


class TestIBP:
    def test_bench(self) -> None:
        out = bench_ibp_bounds(seed=5, n_train=80, n_test=30, iters=40)
        assert 0 <= out["synthetic_ibp_certified_acc"] <= 1


class TestCrown:
    def test_bench(self) -> None:
        out = bench_crown_bound(seed=7, n_train=80, n_test=30, iters=40)
        assert 0 <= out["synthetic_crown_certified_acc"] <= 1


class TestLip:
    def test_bench(self) -> None:
        out = bench_lipschitz_net(seed=9, n_train=80, n_test=30, iters=40)
        assert out["synthetic_lip_empirical_slope"] >= 0


class TestVNN:
    def test_bench(self) -> None:
        out = bench_vector_neurons(seed=11, n_train=80, n_test=30, m_pts=6, iters=40, c_out=8)
        assert 0 <= out["synthetic_vnn_acc_rot"] <= 1


class TestGTK:
    def test_bench(self) -> None:
        out = bench_gumbel_topk(seed=13, n_train=80, m=8, k=2, iters=40)
        assert 0 <= out["synthetic_gtopk_precision"] <= 1
