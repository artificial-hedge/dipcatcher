"""Wave-191 exploration canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.count_bonus import bench_count_bonus
from quant_fund.models.go_explore import bench_go_explore
from quant_fund.models.icm_explore import bench_icm_explore
from quant_fund.models.ngu_explore import bench_ngu_explore
from quant_fund.models.ride_explore import bench_ride_explore
from quant_fund.models.rnd_explore import bench_rnd_explore


class TestCount:
    def test_bench(self) -> None:
        out = bench_count_bonus(seed=3, beta=0.4)
        assert 0.0 <= out["synthetic_count_coverage"] <= 1.0


class TestRND:
    def test_bench(self) -> None:
        out = bench_rnd_explore(seed=5)
        assert np.isfinite(out["synthetic_rnd_coverage"])


class TestICM:
    def test_bench(self) -> None:
        out = bench_icm_explore(seed=7)
        assert np.isfinite(out["synthetic_icm_coverage"])


class TestNGU:
    def test_bench(self) -> None:
        out = bench_ngu_explore(seed=9)
        assert np.isfinite(out["synthetic_ngu_coverage"])


class TestRIDE:
    def test_bench(self) -> None:
        out = bench_ride_explore(seed=11)
        assert np.isfinite(out["synthetic_ride_coverage"])


class TestGoExplore:
    def test_bench(self) -> None:
        out = bench_go_explore(seed=13, rounds=80)
        assert 0.0 <= out["synthetic_goexp_coverage"] <= 1.0
