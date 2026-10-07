"""Curriculum learning — easy→hard ordering (Bengio et al. 2009).

Training sorted by difficulty score (distractor magnitude) vs
anti-curriculum (hard→easy) vs shuffled: curriculum reaches higher
held-out accuracy in few epochs — the canonical CL result.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._data_synth import synth_dataset


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("curriculum_magnitude needs the torch `nn` extra") from exc


def bench_curriculum_magnitude(
    seed: int = 311,
    n: int = 800,
    stages: int = 4,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    x_tr, y_tr = x[:cut], y[:cut]
    x_te, y_te = x[cut:], y[cut:]
    diff = np.abs(x_tr[:, 3:]).sum(-1)  # distractor magnitude = difficulty
    order_easy = diff.argsort()
    order_hard = diff.argsort()[::-1].copy()
    order_rand = np.random.default_rng(seed).permutation(cut)

    torch = _torch()

    def staged(order):
        # online SGD over ordered chunks — ordering changes final weights
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        x_t = torch.tensor(x_tr).float()
        y_t = torch.tensor(y_tr)
        for s in range(stages):
            idx = order[s * cut // stages : (s + 1) * cut // stages]
            for _e in range(5):
                loss = torch.nn.functional.cross_entropy(net(x_t[idx]), y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return float(
                (net(torch.tensor(x_te).float()).argmax(-1) == torch.tensor(y_te)).float().mean()
            )

    acc_c = staged(order_easy)
    acc_a = staged(order_hard)
    acc_r = staged(order_rand)
    return {
        "synthetic_curr_acc": float(acc_c),
        "synthetic_curr_anti_acc": float(acc_a),
        "synthetic_curr_shuffled_acc": float(acc_r),
        "synthetic_curr_gain_vs_anti": float(acc_c - acc_a),
        "synthetic_torch_available": 1.0,
    }
