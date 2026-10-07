"""Meta-learned UQ — MAML-style features + per-task head + learned sigma.

Finn et al. 2017 / Antoniou et al. 2020: a shared feature net is meta-trained
so a few context-point gradient steps specialize it per task; a Gaussian head
predicts (mu, sigma) after adaptation. Bench compares calibrated coverage and
MSE vs a pooled (no-adaptation) regressor. SYNTHETIC only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.neural_process import synth_np_tasks

FloatArray = NDArray[np.float64]

_SEED = 20261005


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_meta_uq(
    seed: int = 19,
    n_train: int = 400,
    n_test: int = 60,
    n_ctx: int = 10,
    n_tgt: int = 50,
    iters: int = 700,
    inner_steps: int = 3,
    d: int = 48,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)

    feat = torch.nn.Sequential(
        torch.nn.Linear(1, d), torch.nn.ReLU(), torch.nn.Linear(d, d), torch.nn.ReLU()
    )
    head = torch.nn.Linear(d, 2)
    opt = torch.optim.Adam(list(feat.parameters()) + list(head.parameters()), lr=1e-3)

    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    xt = torch.tensor(tr[2], dtype=torch.float32)
    yt = torch.tensor(tr[3], dtype=torch.float32)

    # first-order MAML: adapt head weights only on context loss
    for _ in range(iters):
        w = head.weight.detach().clone().requires_grad_(True)
        b = head.bias.detach().clone().requires_grad_(True)
        for _ in range(inner_steps):
            out = torch.nn.functional.linear(feat(xc), w, b)
            loss_c = ((out[..., 0] - yc[..., 0]) ** 2).mean()
            gw, gb = torch.autograd.grad(loss_c, [w, b])
            w = (w - 0.05 * gw).detach().requires_grad_(True)
            b = (b - 0.05 * gb).detach().requires_grad_(True)
        out_t = torch.nn.functional.linear(feat(xt), w, b)
        mu, log_s = out_t[..., 0], out_t[..., 1].clamp(-6, 3)
        nll = (
            0.5 * ((yt[..., 0] - mu) ** 2) / log_s.exp().pow(2) + log_s + 0.5 * np.log(2 * np.pi)
        ).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()

    # evaluate: adapt per test task then predict
    mus, sigs = [], []
    for i in range(n_test):
        xci = torch.tensor(te[0][i], dtype=torch.float32)
        yci = torch.tensor(te[1][i], dtype=torch.float32)
        xti = torch.tensor(te[2][i], dtype=torch.float32)
        w = head.weight.detach().clone().requires_grad_(True)
        b = head.bias.detach().clone().requires_grad_(True)
        for _ in range(inner_steps):
            out = torch.nn.functional.linear(feat(xci), w, b)
            loss_c = ((out[..., 0] - yci[..., 0]) ** 2).mean()
            gw, gb = torch.autograd.grad(loss_c, [w, b])
            w = (w - 0.05 * gw).detach().requires_grad_(True)
            b = (b - 0.05 * gb).detach().requires_grad_(True)
        out_t = torch.nn.functional.linear(feat(xti), w, b)
        mus.append(out_t[..., 0].detach().numpy())
        sigs.append(out_t[..., 1].clamp(-6, 3).exp().detach().numpy())
    mu_arr = np.stack([np.asarray(m) for m in mus])
    sig_arr = np.stack([np.asarray(s) for s in sigs])

    mse_meta = float(np.mean((mu_arr - te[3][..., 0]) ** 2))
    # pooled no-adapt baseline: same net without inner steps
    with torch.no_grad():
        xtb = torch.tensor(te[2], dtype=torch.float32)
        out_b = head(feat(xtb))
        mu_b = out_b[..., 0].numpy()
    mse_pool = float(np.mean((np.asarray(mu_b) - te[3][..., 0]) ** 2))
    cov90 = float(np.mean(np.abs(te[3][..., 0] - mu_arr) <= 1.645 * sig_arr))
    return {
        "synthetic_meta_mse": mse_meta,
        "synthetic_meta_pooled_mse": mse_pool,
        "synthetic_meta_mse_gain": mse_pool - mse_meta,
        "synthetic_meta_cov90": cov90,
        "synthetic_meta_cov90_err": abs(cov90 - 0.90),
        "synthetic_torch_available": 1.0,
    }
