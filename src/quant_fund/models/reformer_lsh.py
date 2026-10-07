"""Reformer-style LSH bucketed attention (Kitaev et al. 2020) (SYNTHETIC).

Angular LSH buckets queries into buckets by hash of a random rotation;
attention only within bucket (sorted order). On the retrieval task LSH
groups the query token near same-key tokens when the random rotation
aligns — an honest approximation trade-off vs full attention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_BUCKETS = 4


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("reformer_lsh needs the torch `nn` extra") from exc


def _lsh_attn(torch, h, seed_mat):
    b, t, d = h.shape
    rot = seed_mat
    score = h @ rot
    bucket = score.argmax(-1)
    order = bucket.argsort(1)
    h_sorted = torch.gather(h, 1, order[:, :, None].expand(-1, -1, d))
    out = torch.zeros_like(h_sorted)
    per = t // _BUCKETS
    for bk in range(_BUCKETS):
        seg = h_sorted[:, bk * per : (bk + 1) * per]
        att = torch.softmax(seg @ seg.transpose(1, 2) / (d**0.5), -1)
        out[:, bk * per : (bk + 1) * per] = att @ seg
    inv = order.argsort(1)
    return torch.gather(out, 1, inv[:, :, None].expand(-1, -1, d))


def bench_reformer_lsh(
    seed: int = 31,
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
    t_len = xtr.shape[1]
    rot = torch.randn(d_h, _BUCKETS)
    proj = torch.nn.Linear(d_in, d_h)
    out = torch.nn.Linear(d_h, nc)
    proj_o = torch.nn.Linear(d_in, d_h)
    out_o = torch.nn.Linear(d_h, nc)
    params = list(proj.parameters()) + list(out.parameters())
    params += list(proj_o.parameters()) + list(out_o.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        h = proj(xtr_t)
        att_l = _lsh_attn(torch, h, rot)
        logits = out(att_l[:, -1])
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
        h = proj(torch.tensor(xte).float())
        acc = (out(_lsh_attn(torch, h, rot)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        h_o = proj_o(torch.tensor(xte).float())
        att_o = torch.softmax(h_o @ h_o.transpose(1, 2) / (d_h**0.5), -1)
        acc_o = (out_o((att_o @ h_o)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
    return {
        "synthetic_lsh_acc": float(acc),
        "synthetic_lsh_full_acc": float(acc_o),
        "synthetic_lsh_acc_gap": float(acc - acc_o),
        "synthetic_lsh_cost_ratio": float(attn_dot_cost(t_len, "lsh", _BUCKETS)),
        "synthetic_torch_available": 1.0,
    }
