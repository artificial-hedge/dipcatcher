"""Wave-161 neural-operator canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._pde_synth import poisson_pairs, rel_l2
from quant_fund.models.cno_lite import bench_cno_lite
from quant_fund.models.deeponet import bench_deeponet
from quant_fund.models.fno_1d import bench_fno_1d
from quant_fund.models.gno_lite import bench_gno_lite
from quant_fund.models.lowrank_op import bench_lowrank_op
from quant_fund.models.pino_residual import bench_pino_residual


class TestPDESynth:
    def test_pairs(self) -> None:
        a_tr, u_tr, a_te, u_te = poisson_pairs(seed=3, n_samp=20, n_grid=16)
        assert a_tr.shape == (10, 16) and u_te.shape == (10, 16)

    def test_rel_l2(self) -> None:
        x = np.ones((2, 4))
        assert rel_l2(x, x) == 0.0


class TestFNO:
    def test_bench(self) -> None:
        out = bench_fno_1d(seed=5, iters=15)
        assert np.isfinite(out["synthetic_fno_rell2"])


class TestDON:
    def test_bench(self) -> None:
        out = bench_deeponet(seed=7, iters=15)
        assert np.isfinite(out["synthetic_don_rell2"])


class TestLO:
    def test_bench(self) -> None:
        out = bench_lowrank_op(seed=9, iters=15)
        assert np.isfinite(out["synthetic_lo_rell2"])


class TestPINO:
    def test_bench(self) -> None:
        out = bench_pino_residual(seed=11, iters=15, n_keep=20)
        assert np.isfinite(out["synthetic_pino_rell2"])


class TestGNO:
    def test_bench(self) -> None:
        out = bench_gno_lite(seed=13, iters=15)
        assert np.isfinite(out["synthetic_gno_rell2"])


class TestCNO:
    def test_bench(self) -> None:
        out = bench_cno_lite(seed=15, iters=15)
        assert np.isfinite(out["synthetic_cno_rell2"])
