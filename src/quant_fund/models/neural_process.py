"""Conditional Neural Process (CNP) for amortized 1-D function regression.

Garnelo et al. 2018: a context set {(x_c, y_c)} is encoded pointwise into
embeddings, pooled by mean, and a decoder maps (x_t, pooled) to a Gaussian
predictive (mu, sigma). Inference is a single forward pass per task — the
amortization win over per-task GP regression. SYNTHETIC bench only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261001


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def synth_np_tasks(
    n_tasks: int, n_ctx: int, n_tgt: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Draw random nonstationary 1-D functions; sample context + target pts.

    Task params: y = sin(a x + phi) * exp(-b x^2) + c x + eps.
    Returns (x_ctx, y_ctx, x_tgt, y_tgt) each (n_tasks, n) or (n_tasks, n, 1).
    """
    a = rng.uniform(1.0, 3.0, n_tasks)
    b = rng.uniform(0.0, 0.4, n_tasks)
    c = rng.uniform(-0.5, 0.5, n_tasks)
    phi = rng.uniform(0, 2 * np.pi, n_tasks)
    x_ctx = rng.uniform(-3.0, 3.0, (n_tasks, n_ctx))
    x_tgt = rng.uniform(-3.0, 3.0, (n_tasks, n_tgt))

    def f(x: FloatArray, i: int) -> FloatArray:
        out: FloatArray = np.asarray(np.sin(a[i] * x + phi[i]) * np.exp(-b[i] * x**2) + c[i] * x)
        return out

    y_ctx = np.stack([f(x_ctx[i], i) for i in range(n_tasks)]) + 0.05 * rng.standard_normal(
        (n_tasks, n_ctx)
    )
    y_tgt = np.stack([f(x_tgt[i], i) for i in range(n_tasks)]) + 0.05 * rng.standard_normal(
        (n_tasks, n_tgt)
    )
    return (
        x_ctx[..., None].astype(np.float64),
        y_ctx[..., None].astype(np.float64),
        x_tgt[..., None].astype(np.float64),
        y_tgt[..., None].astype(np.float64),
    )


def _kernel_ridge(xc: FloatArray, yc: FloatArray, xt: FloatArray, lam: float = 1e-2) -> FloatArray:
    """Per-task RBF kernel-ridge baseline on context points only."""
    n, m = xc.shape[0], xt.shape[1]
    preds = np.zeros((n, m))
    for i in range(n):
        xi = xc[i, :, 0]
        yi = yc[i, :, 0]
        med = np.median(np.abs(xi[:, None] - xi[None, :]) + 1e-9)
        k = np.exp(-((xi[:, None] - xi[None, :]) ** 2) / (2 * med**2 + 1e-9))
        kt = np.exp(-((xt[i, :, 0][:, None] - xi[None, :]) ** 2) / (2 * med**2 + 1e-9))
        alpha = np.linalg.solve(k + lam * np.eye(xi.shape[0]), yi)
        preds[i] = kt @ np.asarray(alpha)
    return preds


def bench_neural_process(
    seed: int = 7,
    n_train: int = 400,
    n_test: int = 60,
    n_ctx: int = 10,
    n_tgt: int = 50,
    iters: int = 900,
    h_dim: int = 48,
    r_dim: int = 48,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)

    enc = torch.nn.Sequential(
        torch.nn.Linear(2, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, r_dim),
    )
    dec = torch.nn.Sequential(
        torch.nn.Linear(r_dim + 1, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, 2),
    )
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    xt = torch.tensor(tr[2], dtype=torch.float32)
    yt = torch.tensor(tr[3], dtype=torch.float32)
    for _ in range(iters):
        h = enc(torch.cat([xc, yc], -1))  # (B, n_ctx, r)
        r = h.mean(1)  # (B, r)
        dec_in = torch.cat([xt, r[:, None, :].expand(-1, n_tgt, -1)], -1)  # (B, n_tgt, r+1)
        out = dec(dec_in)
        mu, log_s = out[..., 0], out[..., 1].clamp(-6, 3)
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
        h = enc(torch.cat([xc_t, yc_t], -1))
        r = h.mean(1)
        out = dec(torch.cat([xt_t, r[:, None, :].expand(-1, n_tgt, -1)], -1))
        mu_t = out[..., 0].numpy()
        sig_t = out[..., 1].clamp(-6, 3).exp().numpy()

    mse_np = float(np.mean((np.asarray(mu_t) - te[3][..., 0]) ** 2))
    base = _kernel_ridge(te[0], te[1], te[2])
    mse_base = float(np.mean((np.asarray(base) - te[3][..., 0]) ** 2))
    cov90 = float(np.mean(np.abs(te[3][..., 0] - np.asarray(mu_t)) <= 1.645 * np.asarray(sig_t)))
    return {
        "synthetic_np_mse": mse_np,
        "synthetic_np_kernel_ridge_mse": mse_base,
        "synthetic_np_mse_gain": mse_base - mse_np,
        "synthetic_np_cov90": cov90,
        "synthetic_np_cov90_err": abs(cov90 - 0.90),
        "torch_available": 1.0,
    }
