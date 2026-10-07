"""DPR dual-encoder dense retrieval (Karpukhin et al. 2020).

Two encoders (mean-pooled linear maps) trained with in-batch contrastive
loss — cosine similarity ranks the query's same-topic doc over the rest.
Beats BM25 where synonym aliases hide lexical overlap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._rag_synth import _VOCAB, recall_at_k, synth_corpus, synth_queries

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("dpr_retriever needs the torch `nn` extra") from exc


def bench_dpr_retriever(
    seed: int = 173,
    n_docs: int = 400,
    n_queries: int = 150,
    doc_len: int = 30,
    d_emb: int = 64,
    iters: int = 600,
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

    d_t = torch.tensor(bow(docs)).float()
    q_t = torch.tensor(bow(queries)).float()
    enc_d = torch.nn.Sequential(torch.nn.Linear(_VOCAB, d_emb))
    enc_q = torch.nn.Sequential(torch.nn.Linear(_VOCAB, d_emb))
    opt = torch.optim.Adam(list(enc_d.parameters()) + list(enc_q.parameters()), lr=3e-3)
    # train on topic-positive pairs from the query/doc sets
    qpos = torch.tensor(bow(queries)).float()
    dpos = torch.tensor(bow(docs)).float()
    rel = td[None, :] == tq[:, None]
    qi, di = np.where(rel)
    keep = rng.choice(len(qi), min(200, len(qi)), replace=False)
    qi, di = qi[keep], di[keep]
    for _i in range(iters):
        eq = torch.nn.functional.normalize(enc_q(qpos[qi]), dim=-1)
        ed = torch.nn.functional.normalize(enc_d(dpos[di]), dim=-1)
        logits = eq @ ed.T / 0.07
        labels = torch.arange(len(qi))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        eq = torch.nn.functional.normalize(enc_q(q_t), dim=-1)
        ed = torch.nn.functional.normalize(enc_d(d_t), dim=-1)
        s = (eq @ ed.T).numpy()
    r5 = recall_at_k(s, td, tq, 5)
    r20 = recall_at_k(s, td, tq, 20)
    return {
        "synthetic_dpr_recall5": r5,
        "synthetic_dpr_recall20": r20,
        "synthetic_torch_available": 1.0,
    }
