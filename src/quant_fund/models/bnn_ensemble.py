"""Deep-ensemble BNN approximation with aleatoric/epistemic split (torch).

M independently-initialized heteroscedastic regression heads trained by
Gaussian NLL form a deep ensemble: predictive variance decomposes into
aleatoric (mean predicted sigma^2) and epistemic (variance of member
means) — the Bayesian proxy that rises out of distribution. Requires the
``nn`` extra; SYNTHETIC heteroscedastic data only.

Bench: in-sample coverage of the 90% predictive interval, and the ratio
of epistemic variance on shifted inputs vs in-distribution inputs —
should exceed 1 substantially.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("bnn_ensemble torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_hetero(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
    """y = sin(2x) + heteroscedastic noise; noise sigma known (bench truth)."""
    x = rng.uniform(-3, 3, n)
    sigma = 0.05 + 0.3 * x**2
    y = np.sin(2 * x) + sigma * rng.standard_normal(n)
    return x[:, None], y, sigma


def _member(torch: Any, seed: int) -> Any:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def bench_bnn_ensemble(seed: int = 77) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n, tr, M = 4000, 3000, 5
    x, y, sigma_true = synth_hetero(n, rng)
    xs = torch.tensor(x, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)

    members = [_member(torch, seed + m) for m in range(M)]
    for net in members:
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        bs = np.asarray(rng.integers(0, tr, 3000))  # bootstrap resample
        for _ in range(300):
            idx = np.asarray(bs[np.asarray(rng.integers(0, len(bs), 256))])
            out = net(xs[idx])
            mu, logv = out[:, 0], out[:, 1].clamp(-6, 4)
            loss = torch.mean(0.5 * torch.exp(-logv) * (ys[idx] - mu) ** 2 + 0.5 * logv)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        outs = torch.stack([net(xs[tr:]) for net in members])  # (M,T,2)
        mus, logv = outs[:, :, 0], outs[:, :, 1]
        alea = float(torch.mean(torch.exp(logv)))
        epi = float(torch.var(mus, dim=0).mean())
        mu_hat = mus.mean(0).numpy()
        sd_hat = np.sqrt(alea + epi)
        yt = y[tr:]
        cover90 = float(np.mean(np.abs(yt - mu_hat) <= 1.645 * sd_hat))
        mae = float(np.mean(np.abs(yt - mu_hat)))
        # OOD epistemic: shifted inputs x in [5,8]
        xo = torch.tensor(rng.uniform(5, 8, 500)[:, None], dtype=torch.float32)
        outs_o = torch.stack([net(xo) for net in members])
        epi_ood = float(torch.var(outs_o[:, :, 0], dim=0).mean())
        # aleatoric calibration: mean predicted sd vs true sigma on test
        sd_corr = float(
            np.corrcoef(np.sqrt(np.mean(np.exp(logv.numpy()), axis=0)), sigma_true[tr:])[0, 1]
        )
    # ridge baseline uncertainty-free
    xr = np.hstack([x, np.ones((n, 1))])
    w = np.asarray(np.linalg.solve(xr[:tr].T @ xr[:tr] + 1e-3 * np.eye(2), xr[:tr].T @ y[:tr]))
    mae_r = float(np.mean(np.abs(xr[tr:] @ w - y[tr:])))
    return {
        "synthetic_bnn_cover90": cover90,
        "synthetic_bnn_mae": mae,
        "synthetic_bnn_ridge_mae": mae_r,
        "synthetic_bnn_mae_margin": mae_r - mae,
        "synthetic_bnn_aleatoric": alea,
        "synthetic_bnn_epistemic_id": epi,
        "synthetic_bnn_epistemic_ood": epi_ood,
        "synthetic_bnn_epistemic_ood_ratio": epi_ood / (epi + 1e-9),
        "synthetic_bnn_sigma_corr": sd_corr,
        "torch_available": 1.0,
    }
