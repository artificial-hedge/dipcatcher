"""Reciprocal rank fusion (Cormack et al. 2009).

score(d) = Σ_l 1/(k + rank_l(d)) — fuses BM25 and dense rankings with no
tuned weights. On the synonym corpus fusion beats either single list.
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
        raise ImportError("rrf_fusion needs the torch `nn` extra") from exc


def bench_rrf_fusion(
    seed: int = 187,
    n_docs: int = 400,
    n_queries: int = 150,
    doc_len: int = 30,
    k: int = 60,
    d_emb: int = 64,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    docs, td = synth_corpus(n_docs, doc_len, rng)
    queries, tq = synth_queries(n_queries, rng)
    s_bm = _bm25_scores(docs, queries)

    def bow(x):
        m = np.zeros((x.shape[0], _VOCAB))
        for i in range(x.shape[0]):
            np.add.at(m[i], x[i], 1.0)
        return m / x.shape[1]

    d_t = torch.tensor(bow(docs)).float()
    q_t = torch.tensor(bow(queries)).float()
    enc = torch.nn.Linear(_VOCAB, d_emb)
    rel = td[None, :] == tq[:, None]
    qi, di = np.where(rel)
    keep = rng.choice(len(qi), min(200, len(qi)), replace=False)
    qi, di = qi[keep], di[keep]
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for _i in range(iters):
        eq = torch.nn.functional.normalize(enc(q_t[qi]), dim=-1)
        ed = torch.nn.functional.normalize(enc(d_t[di]), dim=-1)
        logits = eq @ ed.T / 0.07
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(qi)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        eq = torch.nn.functional.normalize(enc(q_t), dim=-1)
        ed = torch.nn.functional.normalize(enc(d_t), dim=-1)
        s_dn = (eq @ ed.T).numpy()

    def rrf(lists):
        s = np.zeros((n_queries, n_docs))
        for lst in lists:
            rank = lst.argsort(-1).argsort(-1)
            s += 1.0 / (k + (n_docs - rank))
        return s

    s_f = rrf([s_bm, s_dn])
    r_bm = recall_at_k(s_bm, td, tq, 5)
    r_dn = recall_at_k(s_dn, td, tq, 5)
    r_f = recall_at_k(s_f, td, tq, 5)
    return {
        "synthetic_rrf_recall5": r_f,
        "synthetic_rrf_bm25_recall5": r_bm,
        "synthetic_rrf_dense_recall5": r_dn,
        "synthetic_rrf_gain_vs_best": r_f - max(r_bm, r_dn),
        "synthetic_torch_available": 1.0,
    }
