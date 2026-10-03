"""ICM (Pathak et al. 2017): inverse+forward dynamics model; intrinsic
reward = forward-model prediction error on next-state features.
Torch-gated small nets.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import q_learn, state_feat


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("icm_explore requires torch (pip install -e .[nn])") from exc
    return torch


_D2A = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


def bench_icm_explore(seed: int = 2851) -> dict[str, float]:
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    feat = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU())
    inv = torch.nn.Linear(32, 4)
    fwd = torch.nn.Sequential(torch.nn.Linear(20, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    for p in feat.parameters():
        p.requires_grad_(False)
        with torch.no_grad():
            p.uniform_(-0.5, 0.5, generator=g)
    opt = torch.optim.Adam(list(inv.parameters()) + list(fwd.parameters()), lr=1e-3)
    buf: list[tuple[np.ndarray, int, np.ndarray]] = []

    def bonus(s, sp, st, ep, rng):
        a = _D2A[(sp[0] - s[0], sp[1] - s[1])]
        buf.append((state_feat(s), a, state_feat(sp)))
        if len(buf) >= 64 and len(buf) % 8 == 0:
            bs = torch.tensor(np.asarray([b[0] for b in buf[-128:]]), dtype=torch.float32)
            ba = torch.tensor([b[1] for b in buf[-128:]])
            bsp = torch.tensor(np.asarray([b[2] for b in buf[-128:]]), dtype=torch.float32)
            fs, fsp = feat(bs), feat(bsp)
            opt.zero_grad()
            inv_loss = torch.nn.functional.cross_entropy(inv(torch.cat([fs, fsp], -1)), ba)
            fwd_loss = (
                (fwd(torch.cat([fs, torch.nn.functional.one_hot(ba, 4).float()], -1)) - fsp) ** 2
            ).mean()
            (inv_loss + fwd_loss).backward()
            opt.step()
        with torch.no_grad():
            fs = feat(torch.tensor(state_feat(s), dtype=torch.float32))
            fsp = feat(torch.tensor(state_feat(sp), dtype=torch.float32))
            ah = torch.nn.functional.one_hot(torch.tensor(a), 4).float()
            err = ((fwd(torch.cat([fs, ah])) - fsp) ** 2).mean()
            return float(err.clamp(0, 2))

    _, cov, succ = q_learn(bonus, seed=seed)
    _, cov_b, succ_b = q_learn(lambda s, sp, st, ep, rng: 0.0, seed=seed)
    return {
        "synthetic_icm_coverage": float(cov),
        "synthetic_baseline_coverage": float(cov_b),
        "synthetic_icm_success": float(succ),
        "synthetic_baseline_success": float(succ_b),
        "torch_available": 1.0,
    }
