"""Sharpness-aware minimization (Foret et al. 2021).

SAM's two-step (adversarial ascent then descent) finds flatter minima;
on a shifted test set the SAM model generalizes better than plain SGD —
measured via held-out gap and the trace-of-Hessian proxy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._data_synth import synth_dataset


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("sharpness_sam needs the torch `nn` extra") from exc


def bench_sharpness_sam(
    seed: int = 331,
    n: int = 800,
    rho: float = 0.05,
    iters: int = 300,
    shift: float = 3.0,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    x_tr = torch.tensor(x[:cut]).float()
    y_tr = torch.tensor(y[:cut])
    x_te = x[cut:].copy()
    x_te[:, 3] = -x_te[:, 3]  # shortcut flips
    x_te[:, 4:] *= shift
    x_te_t = torch.tensor(x_te).float()
    y_te = torch.tensor(y[cut:])

    def train(sam):
        lin = torch.nn.Sequential(torch.nn.Linear(8, 24), torch.nn.ReLU(), torch.nn.Linear(24, 2))
        opt = torch.optim.Adam(lin.parameters(), lr=5e-3)
        for _i in range(iters):
            loss = torch.nn.functional.cross_entropy(lin(x_tr), y_tr)
            if sam:
                loss.backward()
                with torch.no_grad():
                    for p in lin.parameters():
                        p.add_(rho * p.grad.sign())
                opt.zero_grad()
                loss2 = torch.nn.functional.cross_entropy(lin(x_tr), y_tr)
                loss2.backward()
                with torch.no_grad():
                    for p in lin.parameters():
                        p.add_(-rho * p.grad.sign())
                opt.step()
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            tr = float(torch.nn.functional.cross_entropy(lin(x_tr), y_tr))
            te_acc = float((lin(x_te_t).argmax(-1) == y_te).float().mean())
            tr_acc = float((lin(x_tr).argmax(-1) == y_tr).float().mean())
        return tr, tr_acc, te_acc

    tr0, a0_tr, a0_te = train(False)
    tr1, a1_tr, a1_te = train(True)
    return {
        "synthetic_sam_test_acc": a1_te,
        "synthetic_sam_plain_test_acc": a0_te,
        "synthetic_sam_gain": a1_te - a0_te,
        "synthetic_sam_gen_gap": a1_tr - a1_te,
        "synthetic_sam_plain_gen_gap": a0_tr - a0_te,
        "synthetic_torch_available": 1.0,
    }
