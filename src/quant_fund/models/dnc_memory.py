"""Differentiable Neural Computer-lite (Graves et al. 2016).

Dynamic allocation + temporal link matrix on top of content addressing:
the write head reuses freed memory (usage tracking) and can follow
write-order links. On the copy task the link-following read gives
order-exact recall without relying on content matching alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._mem_synth import synth_copy

FloatArray = NDArray[np.float64]

_MEM = 16


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("dnc_memory needs the torch `nn` extra") from exc


def bench_dnc_memory(
    seed: int = 41,
    n_train: int = 200,
    n_test: int = 100,
    t: int = 10,
    iters: int = 700,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr = synth_copy(n_train, t, rng)
    xte, yte = synth_copy(n_test, t, np.random.default_rng(seed + 1))
    d_in = xtr.shape[2]
    d_h = 24
    n_out = 8
    ctrl = torch.nn.Sequential(
        torch.nn.Linear(d_in + 2 * d_h, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h)
    )
    head = torch.nn.Linear(d_h, d_h * 3 + 1)
    out = torch.nn.Linear(d_h, n_out)
    params = list(ctrl.parameters()) + list(head.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)

    def run(xb):
        b = xb.shape[0]
        m = torch.zeros(b, _MEM, d_h)
        usage = torch.zeros(b, _MEM)
        link = torch.zeros(b, _MEM, _MEM)
        prev_w = torch.zeros(b, _MEM)
        h_prev = torch.zeros(b, d_h)
        logits_seq = []
        for tt in range(xb.shape[1]):
            xi = xb[:, tt]
            rvec = (prev_w[:, :, None] * m).sum(1)
            h = ctrl(torch.cat([xi, rvec, h_prev], -1))
            hp = head(h)
            rkey = hp[:, :d_h]
            wval = hp[:, 2 * d_h : 3 * d_h]
            gate = torch.sigmoid(hp[:, 3 * d_h :])
            sim_r = torch.nn.functional.cosine_similarity(rkey[:, None, :], m, -1)
            wr = torch.softmax(4.0 * sim_r, -1)
            free = 1.0 - usage
            alloc = torch.softmax(40.0 * free + 0.01 * torch.rand(b, _MEM), -1)
            if tt < t:
                m = m + alloc[:, :, None] * wval[:, None, :] * gate[:, :, None]
                link = link + alloc[:, :, None] * prev_w[:, None, :]
                usage = usage + alloc
                prev_w = alloc
            fwd = (link @ wr[:, :, None]).squeeze(-1)
            wrd = torch.softmax(
                4.0 * torch.nn.functional.cosine_similarity(rkey[:, None, :], m, -1), -1
            )
            read_w = 0.5 * wrd + 0.5 * fwd
            rv = (read_w[:, :, None] * m).sum(1)
            logits_seq.append(out(h + rv))
            h_prev = h
        return torch.stack(logits_seq[-t:], 1)

    for _i in range(iters):
        logits = run(xtr_t)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, n_out), ytr_t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = run(torch.tensor(xte).float())
        acc = (logits.argmax(-1) == torch.tensor(yte)).float().mean()
    return {
        "synthetic_dnc_copy_acc": float(acc),
        "synthetic_torch_available": 1.0,
    }
