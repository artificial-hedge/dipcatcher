"""Wave-193 optimal-control canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.ddp_solve import bench_ddp_solve
from quant_fund.models.lqg_control import bench_lqg_control
from quant_fund.models.lqr_control import bench_lqr_control
from quant_fund.models.mpc_qp import bench_mpc_qp
from quant_fund.models.mppi_control import bench_mppi_control
from quant_fund.models.pmp_bangbang import bench_pmp_bangbang


class TestLQR:
    def test_bench(self) -> None:
        out = bench_lqr_control()
        assert out["synthetic_lqr_cost"] < out["synthetic_pd_cost"]


class TestDDP:
    def test_bench(self) -> None:
        out = bench_ddp_solve(iters=15)
        assert np.isfinite(out["synthetic_ddp_cost"])


class TestMPPI:
    def test_bench(self) -> None:
        out = bench_mppi_control(n_samples=32)
        assert np.isfinite(out["synthetic_mppi_cost"])


class TestPMP:
    def test_bench(self) -> None:
        out = bench_pmp_bangbang()
        assert np.isfinite(out["synthetic_bangbang_time"])


class TestMPC:
    def test_bench(self) -> None:
        out = bench_mpc_qp(horizon=12, steps=15)
        assert np.isfinite(out["synthetic_mpc_cost"])


class TestLQG:
    def test_bench(self) -> None:
        out = bench_lqg_control(seed=3)
        assert np.isfinite(out["synthetic_lqg_cost"])
