"""Wave-183 causal-structure-DL canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._cs_synth import sem_data, shd
from quant_fund.models.cam_prune import bench_cam_prune
from quant_fund.models.dag_gnn import bench_dag_gnn
from quant_fund.models.dagma_lin import bench_dagma_lin
from quant_fund.models.golem_ev import bench_golem_ev
from quant_fund.models.notears import bench_notears
from quant_fund.models.notears_mlp import bench_notears_mlp


class TestFixture:
    def test_sem(self) -> None:
        X, B = sem_data(seed=3, n=300)
        assert X.shape == (300, 6)
        assert np.isfinite(X).all()
        assert shd(B, B) == 0


class TestNotears:
    def test_bench(self) -> None:
        out = bench_notears(seed=3, steps=120)
        assert out["synthetic_notears_shd"] >= 0


class TestDagma:
    def test_bench(self) -> None:
        out = bench_dagma_lin(seed=5, steps=120)
        assert out["synthetic_dagma_shd"] >= 0


class TestGolem:
    def test_bench(self) -> None:
        out = bench_golem_ev(seed=7, steps=120)
        assert out["synthetic_golem_shd"] >= 0


class TestNtMlp:
    def test_bench(self) -> None:
        out = bench_notears_mlp(seed=9, steps=120)
        assert out["synthetic_ntmlp_shd"] >= 0


class TestDagGnn:
    def test_bench(self) -> None:
        out = bench_dag_gnn(seed=11, steps=120)
        assert out["synthetic_daggnn_shd"] >= 0


class TestCam:
    def test_bench(self) -> None:
        out = bench_cam_prune(seed=13)
        assert out["synthetic_cam_shd"] >= 0


class TestSEMBound:
    def test_edges_beyond_acyclic_max(self) -> None:
        import pytest

        from quant_fund.models._cs_synth import sem_data

        with pytest.raises(ValueError):
            sem_data(seed=0, d=3, edges=10)
