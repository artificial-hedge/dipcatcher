"""Dynamic-routing capsule layer (Sabour et al. 2017) (SYNTHETIC).

Part capsules → one output capsule via 3 iterations of routing-by-
agreement; on the part-whole fixture the routed agreement beats a mean-
pooled MLP that ignores which parts agree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_partwhole

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("capsule_dynamic needs the torch `nn` extra") from exc


def _squash(v):
    import torch

    n = v.pow(2).sum(-1, keepdim=True)
    return (n / (1 + n)) * v / torch.sqrt(n + 1e-8)


def bench_capsule_dynamic(
    seed: int = 71,
    n_train: int = 400,
    n_test: int = 200,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr = synth_partwhole(n_train, rng)
    xte, yte = synth_partwhole(n_test, np.random.default_rng(seed + 1))
    n_parts, d_part = xtr.shape[1], xtr.shape[2]
    d_out = 8
    w = torch.nn.Parameter(0.2 * torch.randn(n_parts, d_part, d_out))
    out = torch.nn.Linear(d_out, 2)
    flat = torch.nn.Sequential(
        torch.nn.Linear(n_parts * d_part, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    )
    params = [w] + list(out.parameters()) + list(flat.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        u_hat = torch.einsum("npd,pdo->npo", xtr_t, w)
        b = torch.zeros(xtr_t.shape[0], n_parts, 1)
        for _r in range(3):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat).sum(1)
            v = _squash(s)
            b = b + (u_hat * v[:, None, :]).sum(-1, keepdim=True)
        logits = out(v)
        logits_f = flat(xtr_t.reshape(xtr_t.shape[0], -1))
        loss = torch.nn.functional.cross_entropy(logits, ytr_t) + torch.nn.functional.cross_entropy(
            logits_f, ytr_t
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        xb = torch.tensor(xte).float()
        u_hat = torch.einsum("npd,pdo->npo", xb, w)
        b = torch.zeros(xb.shape[0], n_parts, 1)
        for _r in range(3):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat).sum(1)
            v = _squash(s)
            b = b + (u_hat * v[:, None, :]).sum(-1, keepdim=True)
        acc = (out(v).argmax(-1) == torch.tensor(yte)).float().mean()
        acc_f = (flat(xb.reshape(xb.shape[0], -1)).argmax(-1) == torch.tensor(yte)).float().mean()
    return {
        "synthetic_capsule_acc": float(acc),
        "synthetic_capsule_flat_acc": float(acc_f),
        "synthetic_capsule_gain": float(acc - acc_f),
        "synthetic_torch_available": 1.0,
    }
