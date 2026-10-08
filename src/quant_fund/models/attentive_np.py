"""Attentive Neural Process (ANP) — cross-attention from target to context.

Kim et al. 2019: replaces the CNP's mean pooling with scaled dot-product
attention so each target queries the context set directly; fixes the CNP's
underfitting/averaging blur around sharp context structure. SYNTHETIC bench.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.neural_process import _kernel_ridge, synth_np_tasks

FloatArray = NDArray[np.float64]

_SEED = 20261002


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_attentive_np(
    seed: int = 11,
    n_train: int = 400,
    n_test: int = 60,
    n_ctx: int = 10,
    n_tgt: int = 50,
    iters: int = 1100,
    d: int = 48,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)

    key_net = torch.nn.Sequential(torch.nn.Linear(2, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
    val_net = torch.nn.Sequential(torch.nn.Linear(2, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
    qry_net = torch.nn.Linear(1, d)
    dec = torch.nn.Sequential(
        torch.nn.Linear(d + 1, d),
        torch.nn.ReLU(),
        torch.nn.Linear(d, d),
        torch.nn.ReLU(),
        torch.nn.Linear(d, 2),
    )
    params = (
        list(key_net.parameters())
        + list(val_net.parameters())
        + list(qry_net.parameters())
        + list(dec.parameters())
    )
    opt = torch.optim.Adam(params, lr=1e-3)

    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    xt = torch.tensor(tr[2], dtype=torch.float32)
    yt = torch.tensor(tr[3], dtype=torch.float32)

    def forward(xcb: Any, ycb: Any, xtb: Any) -> Any:
        k = key_net(torch.cat([xcb, ycb], -1))  # (B, nc, d)
        v = val_net(torch.cat([xcb, ycb], -1))
        q = qry_net(xtb)  # (B, nt, d)
        att = torch.softmax(q @ k.transpose(1, 2) / np.sqrt(d), -1)  # (B, nt, nc)
        rep = att @ v  # (B, nt, d)
        out = dec(torch.cat([xtb, rep], -1))
        return out[..., 0], out[..., 1].clamp(-6, 3)

    for _ in range(iters):
        mu, log_s = forward(xc, yc, xt)
        nll = (
            0.5 * ((yt[..., 0] - mu) ** 2) / log_s.exp().pow(2) + log_s + 0.5 * np.log(2 * np.pi)
        ).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()

    with torch.no_grad():
        xc_t = torch.tensor(te[0], dtype=torch.float32)
        yc_t = torch.tensor(te[1], dtype=torch.float32)
        xt_t = torch.tensor(te[2], dtype=torch.float32)
        mu_t, sig_t = forward(xc_t, yc_t, xt_t)
        mu_np, sig_np = mu_t.numpy(), sig_t.exp().numpy()

    mse_anp = float(np.mean((np.asarray(mu_np) - te[3][..., 0]) ** 2))
    base = _kernel_ridge(te[0], te[1], te[2])
    mse_base = float(np.mean((np.asarray(base) - te[3][..., 0]) ** 2))
    cov90 = float(np.mean(np.abs(te[3][..., 0] - np.asarray(mu_np)) <= 1.645 * np.asarray(sig_np)))
    return {
        "synthetic_anp_mse": mse_anp,
        "synthetic_anp_kernel_ridge_mse": mse_base,
        "synthetic_anp_mse_gain": mse_base - mse_anp,
        "synthetic_anp_cov90": cov90,
        "synthetic_anp_cov90_err": abs(cov90 - 0.90),
        "synthetic_torch_available": 1.0,
    }
