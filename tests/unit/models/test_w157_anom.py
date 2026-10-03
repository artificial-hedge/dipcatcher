"""Wave-157 anomaly-detection canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import anom_series, auc
from quant_fund.models.anom_transformer import bench_anom_transformer
from quant_fund.models.dagmm import bench_dagmm
from quant_fund.models.deep_svdd import bench_deep_svdd
from quant_fund.models.rrcf import _build, _codisp, bench_rrcf
from quant_fund.models.tranad import bench_tranad
from quant_fund.models.usad import bench_usad


class TestSynth:
    def test_series(self) -> None:
        x, y = anom_series(seed=3, n=80, n_anom=5)
        assert x.shape == (80, 3) and y.sum() >= 5

    def test_auc(self) -> None:
        assert auc(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0


class TestSVDD:
    def test_bench(self) -> None:
        out = bench_deep_svdd(seed=5, iters=15)
        assert 0 <= out["synthetic_svdd_auc"] <= 1


class TestDAGMM:
    def test_bench(self) -> None:
        out = bench_dagmm(seed=7, iters=15)
        assert 0 <= out["synthetic_dagmm_auc"] <= 1


class TestUSAD:
    def test_bench(self) -> None:
        out = bench_usad(seed=9, iters=15)
        assert 0 <= out["synthetic_usad_auc"] <= 1


class TestAT:
    def test_bench(self) -> None:
        out = bench_anom_transformer(seed=11, iters=10)
        assert 0 <= out["synthetic_at_auc"] <= 1


class TestRRCF:
    def test_codisp(self) -> None:
        rng = np.random.default_rng(1)
        root = _build(rng.standard_normal((40, 3)), rng)
        assert _codisp(root, np.zeros(3)) >= 0

    def test_bench(self) -> None:
        out = bench_rrcf(seed=13, n_trees=5)
        assert 0 <= out["synthetic_rrcf_auc"] <= 1


class TestTranAD:
    def test_bench(self) -> None:
        out = bench_tranad(seed=15, iters=10)
        assert 0 <= out["synthetic_tranad_auc"] <= 1
