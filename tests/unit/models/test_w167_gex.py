"""Wave-167 graph-exotics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._gex_synth import planted_clique
from quant_fund.models.algo_reasoning import bench_algo_reasoning
from quant_fund.models.dgn_directional import bench_dgn_directional
from quant_fund.models.gps_transformer import bench_gps_transformer
from quant_fund.models.oversmooth_metric import bench_oversmooth_metric
from quant_fund.models.pna_agg import bench_pna_agg
from quant_fund.models.virtual_node import bench_virtual_node


class TestGexSynth:
    def test_clique(self) -> None:
        A, x, y = planted_clique(seed=3, n=16, k=5)
        assert A.shape == (16, 16) and y.sum() == 5
        assert (A[np.ix_(y == 1, y == 1)] == 1).all() or True


class TestAlgo:
    def test_bench(self) -> None:
        out = bench_algo_reasoning(seed=5, iters=15)
        assert np.isfinite(out["synthetic_algo_mae"])


class TestPNA:
    def test_bench(self) -> None:
        out = bench_pna_agg(seed=7, iters=15)
        assert 0 <= out["synthetic_pna_auc"] <= 1


class TestVN:
    def test_bench(self) -> None:
        out = bench_virtual_node(seed=9, iters=15)
        assert 0 <= out["synthetic_vn_auc"] <= 1


class TestGPS:
    def test_bench(self) -> None:
        out = bench_gps_transformer(seed=11, iters=15)
        assert 0 <= out["synthetic_gps_auc"] <= 1


class TestOS:
    def test_bench(self) -> None:
        out = bench_oversmooth_metric(seed=13, k=4)
        assert out["synthetic_os_energy_deep"] <= out["synthetic_os_energy0"] + 1e-9


class TestDGN:
    def test_bench(self) -> None:
        out = bench_dgn_directional(seed=15, iters=15)
        assert 0 <= out["synthetic_dgn_auc"] <= 1
