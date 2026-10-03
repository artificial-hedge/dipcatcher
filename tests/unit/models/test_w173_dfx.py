"""Wave-173 diffusion-exotics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cold_diffusion import bench_cold_diffusion
from quant_fund.models.ddim_ode import bench_ddim_ode
from quant_fund.models.diff_distill import bench_diff_distill
from quant_fund.models.edm_karras import bench_edm_karras
from quant_fund.models.rectified_flow import bench_rectified_flow
from quant_fund.models.stoch_interp import bench_stoch_interp


class TestEDM:
    def test_bench(self) -> None:
        out = bench_edm_karras(seed=3, iters=60, steps=5)
        assert np.isfinite(out["synthetic_edm_mmd"])


class TestRectFlow:
    def test_bench(self) -> None:
        out = bench_rectified_flow(seed=5, iters=60, steps=5)
        assert np.isfinite(out["synthetic_rf_mmd"])


class TestStochInterp:
    def test_bench(self) -> None:
        out = bench_stoch_interp(seed=7, iters=60, steps=5)
        assert np.isfinite(out["synthetic_si_mmd"])


class TestDDIM:
    def test_bench(self) -> None:
        out = bench_ddim_ode(seed=9, iters=60, steps=5)
        assert np.isfinite(out["synthetic_ddim_mmd"])


class TestCold:
    def test_bench(self) -> None:
        out = bench_cold_diffusion(seed=11, iters=60, steps=5)
        assert np.isfinite(out["synthetic_cold_mmd"])


class TestDistill:
    def test_bench(self) -> None:
        out = bench_diff_distill(seed=13, iters=60, steps=4)
        assert np.isfinite(out["synthetic_dd_student_mmd"])
