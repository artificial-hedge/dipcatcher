"""BM25 sparse retrieval (Robertson & Zaragoza 2009).

TF·IDF with saturation k1 and length norm b over the topic corpus.
Sparse lexical match partially recovers topic relevance; dense methods
close the remainder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._rag_synth import _VOCAB, recall_at_k, synth_corpus, synth_queries

FloatArray = NDArray[np.float64]


def _bm25_scores(
    docs: np.ndarray, queries: np.ndarray, k1: float = 1.2, b: float = 0.75
) -> FloatArray:
    n = docs.shape[0]
    df = np.bincount(docs.ravel(), minlength=_VOCAB) > 0
    df = np.array([(docs == t).any(1).sum() for t in range(_VOCAB)])
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1)
    dl = docs.shape[1]
    avg = dl
    s = np.zeros((queries.shape[0], n))
    for qi, q in enumerate(queries):
        for tok in set(q.tolist()):
            tf = (docs == tok).sum(-1)
            s[qi] += idf[tok] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
    return s


def bench_bm25_retriever(
    seed: int = 171,
    n_docs: int = 400,
    n_queries: int = 150,
    doc_len: int = 30,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    docs, td = synth_corpus(n_docs, doc_len, rng)
    queries, tq = synth_queries(n_queries, rng)
    s = _bm25_scores(docs, queries)
    r5 = recall_at_k(s, td, tq, 5)
    r20 = recall_at_k(s, td, tq, 20)
    rand = rng.random((n_queries, n_docs))
    r5_rand = recall_at_k(rand, td, tq, 5)
    return {
        "synthetic_bm25_recall5": r5,
        "synthetic_bm25_recall20": r20,
        "synthetic_bm25_random_recall5": r5_rand,
        "synthetic_bm25_gain": r5 - r5_rand,
    }
