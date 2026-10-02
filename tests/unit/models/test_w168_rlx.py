"""Wave-168 RL-exotics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.awac import bench_awac
from quant_fund.models.crossq import bench_crossq
from quant_fund.models.dr3_reg import bench_dr3_reg
from quant_fund.models.ob2i import bench_ob2i
from quant_fund.models.redq import bench_redq
from quant_fund.models.td7_lite import bench_td7_lite


class TestAWAC:
    def test_bench(self) -> None:
        out = bench_awac(seed=5, steps=400)
        assert np.isfinite(out["synthetic_awac_mean_reward"])


class TestREDQ:
    def test_bench(self) -> None:
        out = bench_redq(seed=7, steps=400, N=4)
        assert np.isfinite(out["synthetic_redq_mean_reward"])


class TestTD7:
    def test_bench(self) -> None:
        out = bench_td7_lite(seed=9, steps=400)
        assert np.isfinite(out["synthetic_td7_mean_reward"])


class TestCrossQ:
    def test_bench(self) -> None:
        out = bench_crossq(seed=11, steps=400)
        assert np.isfinite(out["synthetic_cq_mean_reward"])


class TestDR3:
    def test_bench(self) -> None:
        out = bench_dr3_reg(seed=13, steps=400)
        assert np.isfinite(out["synthetic_dr3_mean_reward"])


class TestOB2I:
    def test_bench(self) -> None:
        out = bench_ob2i(seed=15, steps=400, N=3)
        assert np.isfinite(out["synthetic_ob2i_mean_reward"])
