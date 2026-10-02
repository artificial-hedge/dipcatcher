"""Wave-178 multi-task-gradient canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cagrad_mtl import bench_cagrad_mtl
from quant_fund.models.gradnorm_bal import bench_gradnorm_bal
from quant_fund.models.imtl_g import bench_imtl_g
from quant_fund.models.mgda_mtl import bench_mgda_mtl
from quant_fund.models.nash_mtl import bench_nash_mtl
from quant_fund.models.pcgrad import bench_pcgrad


class TestPCGrad:
    def test_bench(self) -> None:
        out = bench_pcgrad(seed=3, iters=80)
        assert np.isfinite(out["synthetic_pcgrad_min_gain"])


class TestMGDA:
    def test_bench(self) -> None:
        out = bench_mgda_mtl(seed=5, iters=80)
        assert np.isfinite(out["synthetic_mgda_min_gain"])


class TestCAGrad:
    def test_bench(self) -> None:
        out = bench_cagrad_mtl(seed=7, iters=80)
        assert np.isfinite(out["synthetic_cagrad_min_gain"])


class TestGradNorm:
    def test_bench(self) -> None:
        out = bench_gradnorm_bal(seed=9, iters=80)
        assert np.isfinite(out["synthetic_gradnorm_w1"])


class TestNash:
    def test_bench(self) -> None:
        out = bench_nash_mtl(seed=11, iters=80)
        assert np.isfinite(out["synthetic_nash_min_gain"])


class TestIMTL:
    def test_bench(self) -> None:
        out = bench_imtl_g(seed=13, iters=80)
        assert np.isfinite(out["synthetic_imtl_min_gain"])
