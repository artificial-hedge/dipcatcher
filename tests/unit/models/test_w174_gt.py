"""Wave-174 graph-temporal canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.agcrn import bench_agcrn
from quant_fund.models.astgcn import bench_astgcn
from quant_fund.models.dcrnn_lite import bench_dcrnn_lite
from quant_fund.models.gwnet_lite import bench_gwnet_lite
from quant_fund.models.mtgnn_lite import bench_mtgnn_lite
from quant_fund.models.stgcn_lite import bench_stgcn_lite


class TestDCRNN:
    def test_bench(self) -> None:
        out = bench_dcrnn_lite(seed=3, iters=60)
        assert np.isfinite(out["synthetic_dcrnn_mse"])


class TestSTGCN:
    def test_bench(self) -> None:
        out = bench_stgcn_lite(seed=5, iters=60)
        assert np.isfinite(out["synthetic_stgcn_mse"])


class TestGWNet:
    def test_bench(self) -> None:
        out = bench_gwnet_lite(seed=7, iters=60)
        assert np.isfinite(out["synthetic_gwn_mse"])


class TestASTGCN:
    def test_bench(self) -> None:
        out = bench_astgcn(seed=9, iters=60)
        assert np.isfinite(out["synthetic_astgcn_mse"])


class TestMTGNN:
    def test_bench(self) -> None:
        out = bench_mtgnn_lite(seed=11, iters=60)
        assert np.isfinite(out["synthetic_mtg_mse"])


class TestAGCRN:
    def test_bench(self) -> None:
        out = bench_agcrn(seed=13, iters=60)
        assert np.isfinite(out["synthetic_agcrn_mse"])
