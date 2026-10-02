"""Wave-181 LM-arch-2 canon tests."""

from __future__ import annotations

from quant_fund.models.gated_deltanet import bench_gated_deltanet
from quant_fund.models.longhorn_ssm import bench_longhorn_ssm
from quant_fund.models.mamba2_ssd import bench_mamba2_ssd
from quant_fund.models.rwkv7 import bench_rwkv7
from quant_fund.models.titans_memory import bench_titans_memory
from quant_fund.models.xlstm_mlstm import bench_xlstm_mlstm


class TestMamba2:
    def test_bench(self) -> None:
        out = bench_mamba2_ssd(seed=3, iters=60)
        assert 0.0 <= out["synthetic_mamba2_recall"] <= 1.0


class TestMLSTM:
    def test_bench(self) -> None:
        out = bench_xlstm_mlstm(seed=5, iters=60)
        assert 0.0 <= out["synthetic_mlstm_recall"] <= 1.0


class TestRWKV7:
    def test_bench(self) -> None:
        out = bench_rwkv7(seed=7, iters=60)
        assert 0.0 <= out["synthetic_rwkv7_recall"] <= 1.0


class TestTitans:
    def test_bench(self) -> None:
        out = bench_titans_memory(seed=9, iters=60)
        assert 0.0 <= out["synthetic_titans_recall"] <= 1.0


class TestGDN:
    def test_bench(self) -> None:
        out = bench_gated_deltanet(seed=11, iters=60)
        assert 0.0 <= out["synthetic_gdn_recall"] <= 1.0


class TestLonghorn:
    def test_bench(self) -> None:
        out = bench_longhorn_ssm(seed=13, iters=60)
        assert 0.0 <= out["synthetic_longhorn_recall"] <= 1.0
