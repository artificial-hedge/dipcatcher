"""Memorizing-Transformer-style kNN memory (Wu et al. 2022).

Each token can attend to frozen keys/values from earlier training
examples through an external kNN memory — on the retrieval task the
memory index recalls key→value pairs seen in training instead of
re-deriving them from weights.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_TOPK = 4


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("memorizing_transformer needs the torch `nn` extra") from exc


def bench_memorizing_transformer(
    seed: int = 33,
    n_train: int = 240,
    n_test: int = 120,
    m_pairs: int = 48,
    iters: int = 900,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr = synth_retrieval(n_train, m_pairs, 4, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, 4, np.random.default_rng(seed + 1))
    d_in, d_h, nc = xtr.shape[2], 32, 4
    proj = torch.nn.Linear(d_in, d_h)
    out = torch.nn.Linear(d_h, nc)
    proj_o = torch.nn.Linear(d_in, d_h)
    out_o = torch.nn.Linear(d_h, nc)
    params = list(proj.parameters()) + list(out.parameters())
    params += list(proj_o.parameters()) + list(out_o.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    mem_keys: list | None = None
    mem_vals: list | None = None
    for _i in range(iters):
        h = proj(xtr_t)
        att = torch.softmax(h @ h.transpose(1, 2) / (d_h**0.5), -1)
        ctx = att @ h
        if mem_keys is not None:
            mk = torch.stack(mem_keys)
            mv = torch.stack(mem_vals)
            knn_d = (h[:, -1][:, None, :] - mk).pow(2).sum(-1)
            top = knn_d.topk(_TOPK, largest=False).indices
            mem_ctx = mv[top].mean(1)
            ctx_last = ctx[:, -1] + mem_ctx
        else:
            ctx_last = ctx[:, -1]
        logits = out(ctx_last)
        h_o = proj_o(xtr_t)
        att_o = torch.softmax(h_o @ h_o.transpose(1, 2) / (d_h**0.5), -1)
        logits_o = out_o((att_o @ h_o)[:, -1])
        loss = torch.nn.functional.cross_entropy(logits, ytr_t) + torch.nn.functional.cross_entropy(
            logits_o, ytr_t
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            mem_keys = list(h[:, -1].detach())
            mem_vals = list(h_o[:, -1].detach())
    with torch.no_grad():
        h = proj(torch.tensor(xte).float())
        att = torch.softmax(h @ h.transpose(1, 2) / (d_h**0.5), -1)
        ctx = att @ h
        mk = torch.stack(mem_keys) if mem_keys else torch.zeros(1, d_h)
        mv = torch.stack(mem_vals) if mem_vals else torch.zeros(1, d_h)
        knn_d = (h[:, -1][:, None, :] - mk).pow(2).sum(-1)
        top = knn_d.topk(min(_TOPK, mk.shape[0]), largest=False).indices
        acc = (out(ctx[:, -1] + mv[top].mean(1)).argmax(-1) == torch.tensor(yte)).float().mean()
        h_o = proj_o(torch.tensor(xte).float())
        att_o = torch.softmax(h_o @ h_o.transpose(1, 2) / (d_h**0.5), -1)
        acc_o = (out_o((att_o @ h_o)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
    return {
        "synthetic_memtr_acc": float(acc),
        "synthetic_memtr_full_acc": float(acc_o),
        "synthetic_memtr_acc_gain": float(acc - acc_o),
        "synthetic_memtr_cost_ratio": float(attn_dot_cost(xte.shape[1], "memknn", _TOPK)),
        "synthetic_torch_available": 1.0,
    }
