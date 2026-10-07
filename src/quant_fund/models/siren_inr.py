"""SIREN sinusoidal implicit representation (Sitzmann et al. 2020) (SYNTHETIC).

w0-scaled sin activations fit the field AND its gradient better than a
ReLU MLP of the same size — the INR literature's key differentiator is
gradient fidelity, which we measure explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_field

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("siren_inr needs the torch `nn` extra") from exc


def bench_siren_inr(
    seed: int = 73,
    n_train: int = 500,
    iters: int = 700,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pts, f, grad = synth_field(n_train, rng)
    p_t = torch.tensor(pts).float()
    f_t = torch.tensor(f).float()
    g_t = torch.tensor(grad).float()
    w0 = 12.0

    s1 = torch.nn.Parameter(torch.randn(2, 64) * np.sqrt(1 / 2) * w0 / 10)
    b1 = torch.nn.Parameter(torch.rand(64) * 2 * np.pi - np.pi)
    s2 = torch.nn.Parameter(torch.randn(64, 64) * np.sqrt(6 / 64) / w0)
    b2 = torch.nn.Parameter(torch.zeros(64))
    s3 = torch.nn.Parameter(torch.randn(64, 1) * np.sqrt(6 / 64) / w0)
    b3 = torch.nn.Parameter(torch.zeros(1))
    siren_p = [s1, b1, s2, b2, s3, b3]

    r1 = torch.nn.Parameter(torch.randn(2, 64) * np.sqrt(2 / 2))
    rb1 = torch.nn.Parameter(torch.zeros(64))
    r2 = torch.nn.Parameter(torch.randn(64, 64) * np.sqrt(2 / 64))
    rb2 = torch.nn.Parameter(torch.zeros(64))
    r3 = torch.nn.Parameter(torch.randn(64, 1) * np.sqrt(2 / 64))
    rb3 = torch.nn.Parameter(torch.zeros(1))
    relu_p = [r1, rb1, r2, rb2, r3, rb3]

    def siren_fwd(x):
        return torch.sin(torch.sin(x @ s1 + b1) @ s2 + b2) @ s3 + b3

    def relu_fwd(x):
        return torch.relu(torch.relu(x @ r1 + rb1) @ r2 + rb2) @ r3 + rb3

    opt = torch.optim.Adam(siren_p + relu_p, lr=2e-3)
    for _i in range(iters):
        loss = (siren_fwd(p_t) - f_t).pow(2).mean() + (relu_fwd(p_t) - f_t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse_s = float((siren_fwd(p_t) - f_t).pow(2).mean())
        mse_r = float((relu_fwd(p_t) - f_t).pow(2).mean())
    p_g = p_t.clone().requires_grad_(True)
    gs = torch.autograd.grad(siren_fwd(p_g).sum(), p_g)[0]
    grad_err_s = float((gs - g_t).abs().mean())
    p_r = p_t.clone().requires_grad_(True)
    gr = torch.autograd.grad(relu_fwd(p_r).sum(), p_r)[0]
    grad_err_r = float((gr - g_t).abs().mean())
    return {
        "synthetic_siren_mse": mse_s,
        "synthetic_siren_relu_mse": mse_r,
        "synthetic_siren_grad_err": grad_err_s,
        "synthetic_siren_relu_grad_err": grad_err_r,
        "synthetic_siren_grad_gain": float(grad_err_r - grad_err_s),
        "synthetic_torch_available": 1.0,
    }
