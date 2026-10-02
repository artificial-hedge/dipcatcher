"""Cross-encoder reranker.

First stage (BM25+dense fusion) proposes top-20; a joint scorer over
the query-doc token pair (concat BoW through an interaction MLP)
reranks — finer interaction than dual-encoder dot products. Reports
MRR and recall@5 of the reranked list vs the first stage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._rag_synth import _VOCAB, recall_at_k, synth_corpus, synth_queries
from quant_fund.models.bm25_retriever import _bm25_scores

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("reranker_crossenc needs the torch `nn` extra") from exc


def _mrr(scores: FloatArray, td, tq) -> float:
    order = scores.argsort(-1)[:, ::-1]
    rr = 0.0
    for i in range(order.shape[0]):
        hits = td[order[i]] == tq[i]
        rr += 1.0 / (np.argmax(hits) + 1) if hits.any() else 0.0
    return float(rr / order.shape[0])


def bench_reranker_crossenc(
    seed: int = 183,
    n_docs: int = 400,
    n_queries: int = 150,
    doc_len: int = 30,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    docs, td = synth_corpus(n_docs, doc_len, rng)
    queries, tq = synth_queries(n_queries, rng)

    def bow(x):
        m = np.zeros((x.shape[0], _VOCAB))
        for i in range(x.shape[0]):
            np.add.at(m[i], x[i], 1.0)
        return m / x.shape[1]

    s1 = _bm25_scores(docs, queries)
    s1 = s1 / (s1.max(-1, keepdims=True) + 1e-9)
    # cross-encoder on concatenated BoW pair
    xe = torch.nn.Sequential(
        torch.nn.Linear(2 * _VOCAB, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
    )
    rel = td[None, :] == tq[:, None]
    qi, di = np.where(rel)
    keep = rng.choice(len(qi), min(200, len(qi)), replace=False)
    qi, di = qi[keep], di[keep]
    qB = torch.tensor(bow(queries[qi])).float()
    dB = torch.tensor(bow(docs[di])).float()
    neg_di = rng.integers(0, n_docs, len(qi))
    dN = torch.tensor(bow(docs[neg_di])).float()
    opt = torch.optim.Adam(xe.parameters(), lr=3e-3)
    for _i in range(iters):
        pos = xe(torch.cat([qB, dB], -1)).squeeze(-1)
        neg = xe(torch.cat([qB, dN], -1)).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pos - neg, torch.ones_like(pos))
        opt.zero_grad()
        loss.backward()
        opt.step()
    # rerank first-stage top-20
    top20 = s1.argsort(-1)[:, -20:]
    s2 = np.zeros((n_queries, n_docs))
    qB_all = torch.tensor(bow(queries)).float()
    dB_all = torch.tensor(bow(docs)).float()
    with torch.no_grad():
        for i in range(n_queries):
            idx = torch.tensor(top20[i])
            s2[i, top20[i]] = (
                xe(torch.cat([qB_all[i][None, :].expand(20, -1), dB_all[idx]], -1))
                .squeeze(-1)
                .numpy()
            )
        s2[s2 == 0] = -1e9
    return {
        "synthetic_xenc_mrr": _mrr(s2, td, tq),
        "synthetic_xenc_first_mrr": _mrr(s1, td, tq),
        "synthetic_xenc_mrr_gain": _mrr(s2, td, tq) - _mrr(s1, td, tq),
        "synthetic_xenc_recall5": recall_at_k(s2, td, tq, 5),
        "synthetic_xenc_first_recall5": recall_at_k(s1, td, tq, 5),
        "torch_available": 1.0,
    }
