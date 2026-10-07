"""Neural Turing Machine-lite (Graves et al. 2014) (SYNTHETIC).

Content+location addressing over an external memory matrix: cosine
content weights sharpened by γ, shifted by a convolution head, gated
erase/add writes. On the copy task the external store beats a
flat state encoder that must compress the whole sequence.
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
        raise ImportError("ntm_memory needs the torch `nn` extra") from exc


def _address(torch, key, m, prev_w, beta, g, shift, gamma):
    sim = torch.nn.functional.cosine_similarity(key[:, None, :], m, dim=-1)
    wc = torch.softmax(beta[:, None] * sim, -1)
    wg = g[:, None] * wc + (1 - g[:, None]) * prev_w
    wsh = torch.zeros_like(wg)
    for s_i in (-1, 0, 1):
        wsh = wsh + shift[:, s_i + 1][:, None] * torch.roll(wg, s_i, dims=1)
    return torch.softmax(gamma[:, None] * torch.log(wsh + 1e-8), -1)


def bench_ntm_memory(
    seed: int = 37,
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
    ctrl = torch.nn.Sequential(
        torch.nn.Linear(d_in + 2 * d_h, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h)
    )
    head = torch.nn.Linear(d_h, d_h + 1 + 1 + 3 + 1 + d_h)
    out = torch.nn.Linear(d_h, 8)
    flat = torch.nn.Sequential(
        torch.nn.Linear(d_in * (t + 1), d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h)
    )
    flat_out = torch.nn.Linear(d_h, 8)
    params = (
        list(ctrl.parameters())
        + list(head.parameters())
        + list(out.parameters())
        + list(flat.parameters())
        + list(flat_out.parameters())
    )
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        b = xtr_t.shape[0]
        m = torch.zeros(b, _MEM, d_h)
        w = torch.zeros(b, _MEM)
        w[:, 0] = 1.0
        h_prev = torch.zeros(b, d_h)
        logits_seq = []
        for tt in range(xtr_t.shape[1]):
            xi = xtr_t[:, tt]
            rvec = (w[:, :, None] * m).sum(1)
            h = ctrl(torch.cat([xi, rvec, h_prev], -1))
            hp = head(h)
            key = hp[:, :d_h]
            beta = torch.nn.functional.softplus(hp[:, d_h])
            g = torch.sigmoid(hp[:, d_h + 1])
            shift = torch.softmax(hp[:, d_h + 2 : d_h + 5], -1)
            gamma = 1.0 + torch.nn.functional.softplus(hp[:, d_h + 5])
            erase = torch.sigmoid(hp[:, d_h + 6 :])
            w = _address(torch, key, m, w, beta, g, shift, gamma)
            if tt < t:
                m = m * (1 - w[:, :, None] * erase[:, None, :]) + w[:, :, None] * h[:, None, :]
            logits_seq.append(out(h))
            h_prev = h
        logits = torch.stack(logits_seq[-t:], 1)
        xin = xtr_t[:, : t + 1].reshape(b, -1)
        h_f = flat(xin)
        logits_f = flat_out(h_f)[:, None, :].expand(-1, t, -1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 8), ytr_t.reshape(-1)
        ) + torch.nn.functional.cross_entropy(logits_f.reshape(-1, 8), ytr_t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        b = torch.tensor(xte).float().shape[0]
        xb = torch.tensor(xte).float()
        m = torch.zeros(b, _MEM, d_h)
        w = torch.zeros(b, _MEM)
        w[:, 0] = 1.0
        h_prev = torch.zeros(b, d_h)
        logits_seq = []
        for tt in range(xb.shape[1]):
            xi = xb[:, tt]
            rvec = (w[:, :, None] * m).sum(1)
            h = ctrl(torch.cat([xi, rvec, h_prev], -1))
            hp = head(h)
            key = hp[:, :d_h]
            beta = torch.nn.functional.softplus(hp[:, d_h])
            g = torch.sigmoid(hp[:, d_h + 1])
            shift = torch.softmax(hp[:, d_h + 2 : d_h + 5], -1)
            gamma = 1.0 + torch.nn.functional.softplus(hp[:, d_h + 5])
            erase = torch.sigmoid(hp[:, d_h + 6 :])
            w = _address(torch, key, m, w, beta, g, shift, gamma)
            if tt < t:
                m = m * (1 - w[:, :, None] * erase[:, None, :]) + w[:, :, None] * h[:, None, :]
            logits_seq.append(out(h))
            h_prev = h
        logits = torch.stack(logits_seq[-t:], 1)
        acc = (logits.argmax(-1) == torch.tensor(yte)).float().mean()
        xin = xb[:, : t + 1].reshape(b, -1)
        h_f = flat(xin)
        logits_f = flat_out(h_f)[:, None, :].expand(-1, t, -1)
        acc_f = (logits_f.argmax(-1) == torch.tensor(yte)).float().mean()
    return {
        "synthetic_ntm_copy_acc": float(acc),
        "synthetic_ntm_flat_acc": float(acc_f),
        "synthetic_ntm_acc_gain": float(acc - acc_f),
        "synthetic_torch_available": 1.0,
    }
