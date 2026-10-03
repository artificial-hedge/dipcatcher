"""Wave-142 sequence-model exotics canon tests."""

from __future__ import annotations

from quant_fund.models.delta_net import bench_delta_net
from quant_fund.models.hyena_conv import bench_hyena_conv
from quant_fund.models.mixture_of_depths import bench_mixture_of_depths
from quant_fund.models.retnet_decay import bench_retnet_decay
from quant_fund.models.rwkv_wkv import bench_rwkv_wkv
from quant_fund.models.s4_ssm import bench_s4_ssm


class TestS4:
    def test_bench(self) -> None:
        out = bench_s4_ssm(seed=3, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_s4_acc"] <= 1


class TestRWKV:
    def test_bench(self) -> None:
        out = bench_rwkv_wkv(seed=5, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_rwkv_acc"] <= 1


class TestHyena:
    def test_bench(self) -> None:
        out = bench_hyena_conv(seed=7, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_hyena_acc"] <= 1


class TestRetNet:
    def test_bench(self) -> None:
        out = bench_retnet_decay(seed=9, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_retnet_acc"] <= 1


class TestDelta:
    def test_bench(self) -> None:
        out = bench_delta_net(seed=11, n_train=120, n_test=40, m_pairs=8, iters=30, d_model=8)
        assert 0 <= out["synthetic_delta_acc"] <= 1


class TestMoD:
    def test_bench(self) -> None:
        out = bench_mixture_of_depths(seed=13, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_mod_acc"] <= 1
