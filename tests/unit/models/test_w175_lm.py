"""Wave-175 LM-components canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.alibi_attn import bench_alibi_attn
from quant_fund.models.moe_router import bench_moe_router
from quant_fund.models.mup_init import bench_mup_init
from quant_fund.models.rmsnorm_block import bench_rmsnorm_block
from quant_fund.models.rope_attn import bench_rope_attn
from quant_fund.models.swiglu_ffn import bench_swiglu_ffn


class TestRoPE:
    def test_bench(self) -> None:
        out = bench_rope_attn(seed=3, iters=60)
        assert np.isfinite(out["synthetic_rope_extrap_gain"])


class TestALiBi:
    def test_bench(self) -> None:
        out = bench_alibi_attn(seed=5, iters=60)
        assert np.isfinite(out["synthetic_alibi_extrap_gain"])


class TestSwiGLU:
    def test_bench(self) -> None:
        out = bench_swiglu_ffn(seed=7, iters=80)
        assert np.isfinite(out["synthetic_swiglu_gain"])


class TestRMSNorm:
    def test_bench(self) -> None:
        out = bench_rmsnorm_block(seed=9, iters=80)
        assert out["synthetic_rms_drift"] < out["synthetic_nonorm_drift"]


class TestMoE:
    def test_bench(self) -> None:
        out = bench_moe_router(seed=11, iters=80)
        assert 0.0 <= out["synthetic_moe_expert_balance"] <= 4.0


class TestMuP:
    def test_bench(self) -> None:
        out = bench_mup_init(seed=13, iters=80)
        assert np.isfinite(out["synthetic_mup_width_gap"])
