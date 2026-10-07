"""Label smoothing (Szegedy et al. 2016) (SYNTHETIC).

CE with soft targets y*(1−ε)+ε/K improves calibration: lower ECE on
held-out vs hard-label training, trading a little accuracy — the
canonical regularization result.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._data_synth import synth_dataset


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("label_smoothing needs the torch `nn` extra") from exc


def _ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = probs.max(-1)
    pred = probs.argmax(-1)
    correct = (pred == y).astype(float)
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - correct[m].mean())
    return float(e)


def bench_label_smoothing(
    seed: int = 313,
    n: int = 800,
    eps: float = 0.15,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    x_tr = torch.tensor(x[:cut]).float()
    y_tr = torch.tensor(y[:cut])
    x_te = torch.tensor(x[cut:]).float()
    y_te = y[cut:]

    def train(smooth):
        lin = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        opt = torch.optim.Adam(lin.parameters(), lr=5e-3)
        for _i in range(iters):
            loss = torch.nn.functional.cross_entropy(lin(x_tr), y_tr, label_smoothing=smooth)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return torch.softmax(lin(x_te), -1).numpy()

    p_hard = train(0.0)
    p_soft = train(eps)
    return {
        "synthetic_ls_ece": _ece(p_soft, y_te),
        "synthetic_ls_hard_ece": _ece(p_hard, y_te),
        "synthetic_ls_ece_gain": _ece(p_hard, y_te) - _ece(p_soft, y_te),
        "synthetic_torch_available": 1.0,
    }
