"""HyDE — hypothetical document embeddings (Gao et al. 2023).

A small generator maps the query to a pseudo-document (canonical
tokens), which is embedded instead of the query — bridging the
lexical gap at encode time. Recall@5 lifts over raw dense retrieval.
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
        raise ImportError("hyde_retrieval needs the torch `nn` extra") from exc


def bench_hyde_retrieval(
    seed: int = 181,
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
    enc = torch.nn.Linear(_VOCAB, d_emb)
    rel = td[None, :] == tq[:, None]
    qi, di = np.where(rel)
    keep = rng.choice(len(qi), min(200, len(qi)), replace=False)
    qi, di = qi[keep], di[keep]
    # generator: query BoW → doc BoW (learned synonym bridge)
    gen = torch.nn.Sequential(
        torch.nn.Linear(_VOCAB, 128), torch.nn.ReLU(), torch.nn.Linear(128, _VOCAB)
    )
    opt = torch.optim.Adam(list(enc.parameters()) + list(gen.parameters()), lr=3e-3)
    for _i in range(iters):
        # generator: reconstruct doc bow from query bow
        rec = gen(q_t[qi])
        loss_g = ((rec.sigmoid() - (d_t[di] > 0).float()) ** 2).mean()
        eq = torch.nn.functional.normalize(enc(gen(q_t[qi])), dim=-1)
        ed = torch.nn.functional.normalize(enc(d_t[di]), dim=-1)
        logits = eq @ ed.T / 0.07
        labels = torch.arange(len(qi))
        loss = loss_g + torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ed = torch.nn.functional.normalize(enc(d_t), dim=-1)
        # raw dense: encode query directly
        eq_raw = torch.nn.functional.normalize(enc(q_t), dim=-1)
        s_raw = (eq_raw @ ed.T).numpy()
        # HyDE: encode generated hypothetical doc
        eq_h = torch.nn.functional.normalize(enc(gen(q_t)), dim=-1)
        s_h = (eq_h @ ed.T).numpy()
    return {
        "synthetic_hyde_recall5": recall_at_k(s_h, td, tq, 5),
        "synthetic_hyde_raw_recall5": recall_at_k(s_raw, td, tq, 5),
        "synthetic_hyde_gain": float(recall_at_k(s_h, td, tq, 5) - recall_at_k(s_raw, td, tq, 5)),
        "synthetic_torch_available": 1.0,
    }
