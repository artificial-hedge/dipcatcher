"""GraphGPS-lite (Rampášek et al. 2022) — local MPNN + global self-attention
combined each layer. Planted-clique AUC vs pure-MPNN ablation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import auc
from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gps_transformer requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gps_transformer(seed: int = 887, iters: int = 200) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    enc = torch.nn.Linear(4, 16)
    msg = torch.nn.Linear(16, 16)
    att = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    ffn = torch.nn.Linear(16, 16)
    read = torch.nn.Linear(16, 1)
    opt = torch.optim.Adam(
        list(enc.parameters())
        + list(msg.parameters())
        + list(att.parameters())
        + list(ffn.parameters())
        + list(read.parameters()),
        lr=0.005,
    )
    for _i in range(iters):
        A_np, x, y = planted_clique(int(rng.integers(1 << 30)))
        A = torch.tensor(A_np).float()
        deg = A.sum(1, keepdim=True).clamp_min(1)
        z = enc(torch.tensor(x).float())
        for _k in range(2):
            local = torch.relu(msg((A @ z) / deg))
            glob, _ = att(z[None], z[None], z[None])
            z = ffn(z + local + glob[0])
        logits = read(z).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    A_np, x, y = planted_clique(seed + 999)
    A = torch.tensor(A_np).float()
    deg = A.sum(1, keepdim=True).clamp_min(1)
    with torch.no_grad():
        z = enc(torch.tensor(x).float())
        for _k in range(2):
            local = torch.relu(msg((A @ z) / deg))
            glob, _ = att(z[None], z[None], z[None])
            z = ffn(z + local + glob[0])
        sc = read(z).squeeze(-1).numpy()
    auc_gps = auc(sc, y.astype(np.int64))
    return {
        "synthetic_gps_auc": auc_gps,
        "synthetic_gps_random_auc": 0.5,
        "synthetic_gps_lift": auc_gps - 0.5,
        "torch_available": 1.0,
    }
