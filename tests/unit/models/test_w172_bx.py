"""Wave-172 bandit-exotics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.corrupt_bandit import bench_corrupt_bandit
from quant_fund.models.cucb import bench_cucb
from quant_fund.models.gittins_index import bench_gittins_index
from quant_fund.models.neural_ucb import bench_neural_ucb
from quant_fund.models.psrl import bench_psrl
from quant_fund.models.whittle_restless import bench_whittle_restless


class TestPSRL:
    def test_bench(self) -> None:
        out = bench_psrl(seed=3, T=400)
        assert np.isfinite(out["synthetic_psrl_total_reward"])


class TestGittins:
    def test_bench(self) -> None:
        out = bench_gittins_index(seed=5, T=400)
        assert np.isfinite(out["synthetic_git_expected_reward"])


class TestWhittle:
    def test_bench(self) -> None:
        out = bench_whittle_restless(seed=7, T=400)
        assert np.isfinite(out["synthetic_whi_mean_reward"])


class TestCUCB:
    def test_bench(self) -> None:
        out = bench_cucb(seed=9, T=400)
        assert np.isfinite(out["synthetic_cucb_mean_reward"])


class TestCorrupt:
    def test_bench(self) -> None:
        out = bench_corrupt_bandit(seed=11, T=400)
        assert np.isfinite(out["synthetic_cb_robust_reward"])


class TestNeuralUCB:
    def test_bench(self) -> None:
        out = bench_neural_ucb(seed=13, T=300)
        assert np.isfinite(out["synthetic_nucb_reward_sum"])
