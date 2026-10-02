"""Wave-144 inference-engine canon tests."""

from __future__ import annotations

from quant_fund.models.flash_attn import bench_flash_attn
from quant_fund.models.gqa_attn import bench_gqa_attn
from quant_fund.models.paged_kv_cache import bench_paged_kv_cache
from quant_fund.models.ring_attn import bench_ring_attn
from quant_fund.models.sliding_window_cache import bench_sliding_window_cache
from quant_fund.models.speculative_decoding import bench_speculative_decoding


class TestSpec:
    def test_bench(self) -> None:
        out = bench_speculative_decoding(seed=3, n_seq=40)
        assert 0 <= out["synthetic_spec_accept_rate"] <= 1


class TestPaged:
    def test_bench(self) -> None:
        out = bench_paged_kv_cache(seed=5, n_seq=10, max_len=32)
        assert 0 <= out["synthetic_paged_waste_frac"] <= 1


class TestFlash:
    def test_bench(self) -> None:
        out = bench_flash_attn(seed=7, t=64, d=16)
        assert out["synthetic_flash_max_err"] < 1e-8


class TestGQA:
    def test_bench(self) -> None:
        out = bench_gqa_attn(seed=9, n_train=120, n_test=40, m_pairs=8, iters=30)
        assert 0 <= out["synthetic_gqa_acc"] <= 1


class TestSWC:
    def test_bench(self) -> None:
        out = bench_sliding_window_cache(seed=11, n_seq=60, t=32, window=8)
        assert 0 <= out["synthetic_swc_mem_frac"] <= 1


class TestRing:
    def test_bench(self) -> None:
        out = bench_ring_attn(seed=13, t=64, d=16)
        assert out["synthetic_ring_max_err"] < 1e-8
