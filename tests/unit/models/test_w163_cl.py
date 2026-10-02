"""Wave-163 lifelong-CL canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.agem_cl import bench_agem_cl
from quant_fund.models.continual_learning import regime_panel
from quant_fund.models.der_cl import bench_der_cl
from quant_fund.models.hat_cl import bench_hat_cl
from quant_fund.models.lwf_cl import bench_lwf_cl
from quant_fund.models.packnet_cl import bench_packnet_cl
from quant_fund.models.piggyback_cl import bench_piggyback_cl


class TestCLFixture:
    def test_regime_panel(self) -> None:
        rng = np.random.default_rng(3)
        x, y = regime_panel("momentum", 32, rng)
        assert x.shape == (32, 4) and y.shape == (32,)


class TestPackNet:
    def test_bench(self) -> None:
        out = bench_packnet_cl(seed=5, T=80)
        assert np.isfinite(out["synthetic_pn_task1_mse"])


class TestLwF:
    def test_bench(self) -> None:
        out = bench_lwf_cl(seed=7, T=80)
        assert np.isfinite(out["synthetic_lwf_task1_mse"])


class TestDER:
    def test_bench(self) -> None:
        out = bench_der_cl(seed=9, T=80)
        assert np.isfinite(out["synthetic_der_task1_mse"])


class TestAGEM:
    def test_bench(self) -> None:
        out = bench_agem_cl(seed=11, T=80)
        assert np.isfinite(out["synthetic_agem_task1_mse"])


class TestPiggyback:
    def test_bench(self) -> None:
        out = bench_piggyback_cl(seed=13, T=80)
        assert np.isfinite(out["synthetic_pb_task1_mse"])


class TestHAT:
    def test_bench(self) -> None:
        out = bench_hat_cl(seed=15, T=80)
        assert np.isfinite(out["synthetic_hat_task1_mse"])
