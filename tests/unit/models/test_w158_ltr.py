"""Wave-158 learning-to-rank canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._ltr_synth import ltr_data, ndcg_at
from quant_fund.models.approx_ndcg_ltr import bench_approx_ndcg_ltr
from quant_fund.models.lambdarank_ltr import bench_lambdarank_ltr
from quant_fund.models.listmle_ltr import bench_listmle_ltr
from quant_fund.models.listnet_ltr import bench_listnet_ltr
from quant_fund.models.neural_sort_ltr import bench_neural_sort_ltr
from quant_fund.models.ranknet_ltr import bench_ranknet_ltr


class TestLTRSynth:
    def test_data(self) -> None:
        x_tr, y_tr, x_te, y_te = ltr_data(seed=3, n_q=8, n_doc=6, d=8)
        assert x_tr.shape == (8, 6, 8) and y_te.shape == (8, 6)

    def test_ndcg_perfect(self) -> None:
        y = np.array([[0, 1, 2]])
        assert ndcg_at(y, y.astype(float), k=3) == 1.0


class TestRankNet:
    def test_bench(self) -> None:
        out = bench_ranknet_ltr(seed=5, iters=15)
        assert 0 <= out["synthetic_ranknet_ndcg10"] <= 1


class TestListNet:
    def test_bench(self) -> None:
        out = bench_listnet_ltr(seed=7, iters=15)
        assert 0 <= out["synthetic_listnet_ndcg10"] <= 1


class TestListMLE:
    def test_bench(self) -> None:
        out = bench_listmle_ltr(seed=9, iters=10)
        assert 0 <= out["synthetic_listmle_ndcg10"] <= 1


class TestLambdaRank:
    def test_bench(self) -> None:
        out = bench_lambdarank_ltr(seed=11, iters=15)
        assert 0 <= out["synthetic_lambda_ndcg10"] <= 1


class TestApproxNDCG:
    def test_bench(self) -> None:
        out = bench_approx_ndcg_ltr(seed=13, iters=15)
        assert 0 <= out["synthetic_andcg_ndcg10"] <= 1


class TestNeuralSort:
    def test_bench(self) -> None:
        out = bench_neural_sort_ltr(seed=15, iters=15)
        assert 0 <= out["synthetic_nsort_ndcg10"] <= 1
