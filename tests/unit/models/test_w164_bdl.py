"""Wave-164 Bayesian-DL canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss
from quant_fund.models.bbb_vi import bench_bbb_vi
from quant_fund.models.concrete_dropout import bench_concrete_dropout
from quant_fund.models.mc_dropout import bench_mc_dropout
from quant_fund.models.snapshot_ens import bench_snapshot_ens
from quant_fund.models.swag_diag import bench_swag_diag
from quant_fund.models.vcl_online import bench_vcl_online


class TestBDLSynth:
    def test_data(self) -> None:
        x, y, xt, yt, xo = bdl_data(seed=3, n=32)
        assert x.shape == (32, 4) and xo.shape[0] == 8

    def test_nll_cov(self) -> None:
        y = np.array([0.0, 1.0])
        assert nll_gauss(y, np.zeros(2), np.ones(2)) > 0
        assert coverage(y, np.zeros(2), np.ones(2)) == 1.0
        assert coverage(y, np.zeros(2), np.full(2, 0.1)) == 0.5


class TestSWAG:
    def test_bench(self) -> None:
        out = bench_swag_diag(seed=5, iters=40, collect=12)
        assert np.isfinite(out["synthetic_swag_nll"])


class TestMCDO:
    def test_bench(self) -> None:
        out = bench_mc_dropout(seed=7, iters=40, T=5)
        assert np.isfinite(out["synthetic_mcdo_nll"])


class TestBBB:
    def test_bench(self) -> None:
        out = bench_bbb_vi(seed=9, iters=40, T=5)
        assert np.isfinite(out["synthetic_bbb_nll"])


class TestSnapshot:
    def test_bench(self) -> None:
        out = bench_snapshot_ens(seed=11, cycles=2, cycle_len=20)
        assert out["synthetic_se_n_members"] == 2.0


class TestConcreteDO:
    def test_bench(self) -> None:
        out = bench_concrete_dropout(seed=13, iters=40, T=5)
        assert 0 < out["synthetic_cd_learned_p"] < 1


class TestVCL:
    def test_bench(self) -> None:
        out = bench_vcl_online(seed=15, T=80)
        assert np.isfinite(out["synthetic_vcl_task1_mse"])
