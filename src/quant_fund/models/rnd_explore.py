"""RND (Burda et al. 2019): fixed random target network vs trained
predictor; intrinsic reward = feature prediction error. Torch-gated
MLP pair on state features.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import q_learn, state_feat


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("rnd_explore requires torch (pip install -e .[nn])") from exc
    return torch


def bench_rnd_explore(seed: int = 2847) -> dict[str, float]:
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    target = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    pred = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    with torch.no_grad():
        for p in target.parameters():
            p.uniform_(-1, 1, generator=g)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    buf: list[np.ndarray] = []

    def bonus(s, sp, st, ep, rng):
        buf.append(state_feat(sp))
        if len(buf) >= 64 and len(buf) % 8 == 0:
            X = torch.tensor(np.asarray(buf[-128:]), dtype=torch.float32)
            opt.zero_grad()
            err = ((pred(X) - target(X)) ** 2).mean()
            err.backward()
            opt.step()
        with torch.no_grad():
            x = torch.tensor(state_feat(sp), dtype=torch.float32)
            return float(((pred(x) - target(x)) ** 2).mean().clamp(0, 2))

    _, cov, succ = q_learn(bonus, seed=seed)
    _, cov_b, succ_b = q_learn(lambda s, sp, st, ep, rng: 0.0, seed=seed)
    return {
        "synthetic_rnd_coverage": float(cov),
        "synthetic_baseline_coverage": float(cov_b),
        "synthetic_rnd_success": float(succ),
        "synthetic_baseline_success": float(succ_b),
        "synthetic_torch_available": 1.0,
    }
