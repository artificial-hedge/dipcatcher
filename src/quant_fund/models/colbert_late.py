"""ColBERT late interaction (Khattab & Zaharia 2020).

Per-token encodings; score = Σ_q max_d cos(q_i, d_j) — MaxSim over
token pairs is finer-grained than DPR's single dot product. On the
topic corpus it lifts recall over the dual encoder.
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
        raise ImportError("colbert_late needs the torch `nn` extra") from exc


def bench_colbert_late(
    seed: int = 179,
    n_docs: int = 400,
    n_queries: int = 150,
    doc_len: int = 30,
    d_emb: int = 32,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    docs, td = synth_corpus(n_docs, doc_len, rng)
    queries, tq = synth_queries(n_queries, rng)
    emb = torch.nn.Embedding(_VOCAB, d_emb)
    torch.nn.init.normal_(emb.weight, std=0.3)
    rel = td[None, :] == tq[:, None]
    qi, di = np.where(rel)
    keep = rng.choice(len(qi), min(200, len(qi)), replace=False)
    qi, di = qi[keep], di[keep]
    opt = torch.optim.Adam(emb.parameters(), lr=5e-3)
    d_t = torch.tensor(docs[di])
    q_t = torch.tensor(queries[qi])
    for _i in range(iters):
        eq = torch.nn.functional.normalize(emb(q_t), dim=-1)
        ed = torch.nn.functional.normalize(emb(d_t), dim=-1)
        gold = torch.nn.functional.normalize(emb(torch.tensor(docs[di])), dim=-1)
        logits = torch.einsum("nqd,mtd->nmqt", eq, gold).max(-1).values.sum(-1) / 0.05
        labels = torch.arange(len(qi))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        eq = torch.nn.functional.normalize(emb(torch.tensor(queries)), dim=-1)
        ed = torch.nn.functional.normalize(emb(torch.tensor(docs)), dim=-1)
        s = torch.einsum("nqd,mtd->nmqt", eq, ed).max(-1).values.sum(-1).numpy()
    r5 = recall_at_k(s, td, tq, 5)
    r20 = recall_at_k(s, td, tq, 20)
    return {
        "synthetic_colbert_recall5": r5,
        "synthetic_colbert_recall20": r20,
        "torch_available": 1.0,
    }
