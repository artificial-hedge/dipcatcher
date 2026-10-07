"""TransE knowledge-graph embeddings (Bordes et al. 2013).

Synthetic typed-entity graph: 8 latent types share a 4-token signature;
relations compose via translations. Trained by margin ranking on
corrupted triples; hits@10 on held-out triples vs random baseline.
"""

from __future__ import annotations

import numpy as np

_NTYPE = 8
_NENT = 40


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("knowledge_graph_embed needs the torch `nn` extra") from exc


def _synth_graph(n_triples: int, rng: np.random.Generator):
    """Entities 0..39 with latent type; relations r: h-type determines t-type."""
    types = rng.integers(0, _NTYPE, _NENT)
    rel_map = rng.permutation(_NTYPE)  # relation 0: t_type = perm(h_type)
    h = rng.integers(0, _NENT, n_triples)
    r = rng.integers(0, 2, n_triples)
    t = np.zeros(n_triples, dtype=np.int64)
    for i in range(n_triples):
        if r[i] == 0:
            cand = np.where(types == rel_map[types[h[i]]])[0]
        else:
            cand = np.where(types == types[h[i]])[0]  # same-type relation
        t[i] = rng.choice(cand)
    return types, h, r, t


def bench_knowledge_graph_embed(
    seed: int = 223,
    n_triples: int = 400,
    n_neg: int = 5,
    d: int = 32,
    iters: int = 300,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    types, h, r, t = _synth_graph(n_triples, rng)
    cut = int(0.8 * n_triples)
    h_tr, r_tr, t_tr = h[:cut], r[:cut], t[:cut]
    _h_te, r_te, t_te = h[cut:], r[cut:], t[cut:]

    E = torch.nn.Embedding(_NENT, d)
    R = torch.nn.Embedding(2, d)
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=5e-3)

    def score(hh, rr, tt):
        return -(E(hh) + R(rr) - E(tt)).norm(dim=-1)

    for _i in range(iters):
        pos = score(torch.tensor(h_tr), torch.tensor(r_tr), torch.tensor(t_tr))
        tn = rng.integers(0, _NENT, len(t_tr))
        neg = score(torch.tensor(h_tr), torch.tensor(r_tr), torch.tensor(tn))
        loss = torch.nn.functional.margin_ranking_loss(pos, neg, torch.ones_like(pos), margin=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    hits = 0
    with torch.no_grad():
        emb = E(torch.arange(_NENT))
        for i in range(len(t_te)):
            s = -(emb + R(torch.tensor(r_te[i])) - emb[t_te[i]]).norm(dim=-1)
            rank = int((s > s[t_te[i]]).sum()) + 1
            hits += rank <= 10
    rand_hit = 10.0 / _NENT
    return {
        "synthetic_kge_hits10": hits / len(t_te),
        "synthetic_kge_random_hits10": rand_hit,
        "synthetic_kge_gain": hits / len(t_te) - rand_hit,
        "synthetic_torch_available": 1.0,
    }
