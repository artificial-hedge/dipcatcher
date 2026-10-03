"""Wave-145 retrieval canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._rag_synth import recall_at_k, synth_corpus, synth_queries
from quant_fund.models.bm25_retriever import bench_bm25_retriever
from quant_fund.models.colbert_late import bench_colbert_late
from quant_fund.models.dpr_retriever import bench_dpr_retriever
from quant_fund.models.hyde_retrieval import bench_hyde_retrieval
from quant_fund.models.reranker_crossenc import bench_reranker_crossenc
from quant_fund.models.rrf_fusion import bench_rrf_fusion


class TestFixture:
    def test_corpus(self) -> None:
        d, t = synth_corpus(30, 20, np.random.default_rng(0))
        assert d.shape == (30, 20) and t.max() < 8

    def test_queries(self) -> None:
        q, t = synth_queries(30, np.random.default_rng(0))
        assert q.shape == (30, 4)

    def test_recall(self) -> None:
        s = np.random.default_rng(0).random((10, 20))
        td = np.arange(20) % 8
        tq = np.arange(10) % 8
        assert 0 <= recall_at_k(s, td, tq, 3) <= 1


class TestBM25:
    def test_bench(self) -> None:
        out = bench_bm25_retriever(seed=3, n_docs=80, n_queries=30, doc_len=20)
        assert 0 <= out["synthetic_bm25_recall5"] <= 1


class TestDPR:
    def test_bench(self) -> None:
        out = bench_dpr_retriever(seed=5, n_docs=80, n_queries=30, doc_len=20, iters=60)
        assert 0 <= out["synthetic_dpr_recall5"] <= 1


class TestColBERT:
    def test_bench(self) -> None:
        out = bench_colbert_late(seed=7, n_docs=80, n_queries=30, doc_len=20, iters=60)
        assert 0 <= out["synthetic_colbert_recall5"] <= 1


class TestHyDE:
    def test_bench(self) -> None:
        out = bench_hyde_retrieval(seed=9, n_docs=80, n_queries=30, doc_len=20, iters=60)
        assert 0 <= out["synthetic_hyde_recall5"] <= 1


class TestXEnc:
    def test_bench(self) -> None:
        out = bench_reranker_crossenc(seed=11, n_docs=80, n_queries=30, doc_len=20, iters=60)
        assert 0 <= out["synthetic_xenc_mrr"] <= 1


class TestRRF:
    def test_bench(self) -> None:
        out = bench_rrf_fusion(seed=13, n_docs=80, n_queries=30, doc_len=20, iters=60)
        assert 0 <= out["synthetic_rrf_recall5"] <= 1
