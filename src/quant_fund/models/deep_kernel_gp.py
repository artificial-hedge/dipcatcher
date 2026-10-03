"""Deep Kernel Learning GP — NN feature map inside an RBF GP.

Wilson et al. 2016: k(x, x') = exp(-||phi(x) - phi(x')||^2 / (2 l^2)) with phi a
learned feature net; the whole stack trains end-to-end by GP marginal
likelihood, so the kernel adapts to nonstationary structure a raw RBF cannot
represent. Closed-form GP posterior at predict time. SYNTHETIC bench.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.neural_process import _kernel_ridge, synth_np_tasks

FloatArray = NDArray[np.float64]

_SEED = 20261003


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_deep_kernel_gp(
    seed: int = 13,
    n_train: int = 300,
    n_test: int = 50,
    n_ctx: int = 14,
    n_tgt: int = 40,
    iters: int = 700,
    d: int = 16,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)

    feat = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, d),
    )
    log_ls = torch.nn.Parameter(torch.zeros(1))
    log_sf = torch.nn.Parameter(torch.zeros(1))
    log_sn = torch.nn.Parameter(torch.tensor(-2.0))
    opt = torch.optim.Adam(list(feat.parameters()) + [log_ls, log_sf, log_sn], lr=2e-3)

    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    xt = torch.tensor(tr[2], dtype=torch.float32)
    yt = torch.tensor(tr[3], dtype=torch.float32)
    n_all = n_ctx + n_tgt

    for _ in range(iters):
        x_all = torch.cat([xc, xt], 1)  # (B, n, 1)
        y_all = torch.cat([yc, yt], 1)[..., 0]  # (B, n)
        z = feat(x_all)  # (B, n, d)
        d2 = ((z[:, :, None, :] - z[:, None, :, :]) ** 2).sum(-1)
        k = log_sf.exp().pow(2) * torch.exp(-d2 / (2 * log_ls.exp().pow(2)))
        k = k + (log_sn.exp().pow(2) + 1e-4) * torch.eye(n_all)
        # batched marginal log-likelihood via Cholesky
        Lc = torch.linalg.cholesky(k)
        alpha = torch.cholesky_solve(y_all[..., None], Lc)[..., 0]
        mll = -(
            -0.5 * (y_all * alpha).sum(-1)
            - Lc.diagonal(dim1=1, dim2=2).log().sum(-1)
            - 0.5 * n_all * np.log(2 * np.pi)
        ).mean()
        opt.zero_grad()
        mll.backward()
        opt.step()

    # closed-form posterior on test tasks
    with torch.no_grad():
        xc_t = torch.tensor(te[0], dtype=torch.float32)
        yc_t = torch.tensor(te[1], dtype=torch.float32)[..., 0]
        xt_t = torch.tensor(te[2], dtype=torch.float32)
        zc = feat(xc_t)
        zt = feat(xt_t)
        kcc = log_sf.exp().pow(2) * torch.exp(
            -((zc[:, :, None, :] - zc[:, None, :, :]) ** 2).sum(-1) / (2 * log_ls.exp().pow(2))
        ) + (log_sn.exp().pow(2) + 1e-4) * torch.eye(n_ctx)
        ktc = log_sf.exp().pow(2) * torch.exp(
            -((zt[:, :, None, :] - zc[:, None, :, :]) ** 2).sum(-1) / (2 * log_ls.exp().pow(2))
        )
        mu = (ktc @ torch.linalg.solve(kcc, yc_t[..., None]))[..., 0].numpy()
        ktt = log_sf.exp().pow(2)
        kct = ktc.transpose(1, 2)
        solved = torch.linalg.solve(kcc, kct).transpose(1, 2)  # (B, nt, nc)
        var = (ktt - (ktc * solved).sum(-1)).clamp(1e-6, None).numpy()

    mse_dk = float(np.mean((np.asarray(mu) - te[3][..., 0]) ** 2))
    base = _kernel_ridge(te[0], te[1], te[2])
    mse_base = float(np.mean((np.asarray(base) - te[3][..., 0]) ** 2))
    cov90 = float(
        np.mean(np.abs(te[3][..., 0] - np.asarray(mu)) <= 1.645 * np.sqrt(np.asarray(var)))
    )
    return {
        "synthetic_dkgp_mse": mse_dk,
        "synthetic_dkgp_kernel_ridge_mse": mse_base,
        "synthetic_dkgp_mse_gain": mse_base - mse_dk,
        "synthetic_dkgp_cov90": cov90,
        "synthetic_dkgp_cov90_err": abs(cov90 - 0.90),
        "torch_available": 1.0,
    }
