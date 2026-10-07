"""Mixup / CutMix augmentation (Zhang et al. 2018, Yun et al. 2019) (SYNTHETIC).

Interpolated (λx + (1−λ)x', λy + (1−λ)y') samples regularize the
decision boundary; on held-out inputs with amplified distractor noise
(shifted test set), mixup-trained model keeps higher accuracy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._data_synth import synth_dataset


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("mixup_cutmix needs the torch `nn` extra") from exc


def bench_mixup_cutmix(
    seed: int = 317,
    n: int = 800,
    alpha: float = 0.2,
    iters: int = 600,
    shift: float = 4.0,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    x_tr = torch.tensor(x[:cut]).float()
    y_tr = torch.tensor(y[:cut])
    x_te = x[cut:].copy()
    x_te[:, 3] = -x_te[:, 3]  # shortcut correlation flips at test time
    x_te[:, 4:] *= shift
    x_te_t = torch.tensor(x_te).float()
    y_te = torch.tensor(y[cut:])

    def train(mixup):
        lin = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        opt = torch.optim.Adam(lin.parameters(), lr=5e-3)
        for _i in range(iters):
            if mixup:
                perm = torch.randperm(len(x_tr))
                lam = float(rng.beta(alpha, alpha))
                xb = lam * x_tr + (1 - lam) * x_tr[perm]
                out = lin(xb)
                loss = lam * torch.nn.functional.cross_entropy(out, y_tr, reduction="none") + (
                    1 - lam
                ) * torch.nn.functional.cross_entropy(out, y_tr[perm], reduction="none")
                loss = loss.mean()
            else:
                loss = torch.nn.functional.cross_entropy(lin(x_tr), y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float((lin(x_te_t).argmax(-1) == y_te).float().mean())

    acc_plain = train(False)
    acc_mix = train(True)
    return {
        "synthetic_mixup_acc": acc_mix,
        "synthetic_mixup_plain_acc": acc_plain,
        "synthetic_mixup_gain": acc_mix - acc_plain,
        "synthetic_torch_available": 1.0,
    }
