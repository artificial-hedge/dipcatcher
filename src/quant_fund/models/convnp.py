"""Convolutional NP (ConvCNP-lite) — translation-equivariant context smoothing.

Gordon et al. 2020: context points are projected onto a fixed grid through a
RBF kernel (density channel + data channel), a CNN refines the grid, and each
target reads off the grid at its location — giving translation equivariance a
pooled CNP lacks. SYNTHETIC bench.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.neural_process import _kernel_ridge, synth_np_tasks

FloatArray = NDArray[np.float64]

_SEED = 20261004

_X_LO, _X_HI, _GRID = -3.5, 3.5, 64


def _torch() -> Any:
    import torch

    _ = torch.nn.Conv1d  # torch + nn extra required
    return torch


def _grid_encode(xc: Any, yc: Any, xs: Any, torch: Any, sigma: float = 0.1) -> Any:
    """Nadaraya-Watson projection of context (x,y) onto a fixed grid."""
    w = torch.exp(-((xs[None, :, None] - xc.transpose(1, 2)) ** 2) / (2 * sigma**2))
    # xc: (B, nc, 1) -> transpose to (B, 1, nc); xs: (G,)
    w = torch.exp(-((xs[None, None, :] - xc) ** 2) / (2 * sigma**2))  # (B, nc, G)
    dens = w.sum(1)  # (B, G)
    data = (w * yc).sum(1) / dens.clamp(1e-6, None)  # (B, G)
    return torch.stack([data, dens], 1)  # (B, 2, G)


def bench_convnp(
    seed: int = 17,
    n_train: int = 400,
    n_test: int = 60,
    n_ctx: int = 10,
    n_tgt: int = 50,
    iters: int = 900,
    ch: int = 24,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)
    xs_grid = torch.linspace(_X_LO, _X_HI, _GRID)

    cnn = torch.nn.Sequential(
        torch.nn.Conv1d(2, ch, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(ch, ch, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(ch, ch, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(ch, 2, 1),
    )
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    xt = torch.tensor(tr[2], dtype=torch.float32)
    yt = torch.tensor(tr[3], dtype=torch.float32)

    def read_off(grid_out: Any, xtb: Any) -> tuple[Any, Any]:
        # linear interp of grid channels at target x positions
        pos = ((xtb[..., 0] - _X_LO) / (_X_HI - _X_LO) * (_GRID - 1)).clamp(0, _GRID - 1.001)
        i0 = pos.long()
        frac = pos - i0.float()
        idx0 = i0[:, None, :].expand(-1, grid_out.shape[1], -1)
        i1 = (i0 + 1).clamp(max=_GRID - 1)
        idx1 = i1[:, None, :].expand(-1, grid_out.shape[1], -1)
        v0 = torch.gather(grid_out, 2, idx0)  # (B, 2, nt)
        v1 = torch.gather(grid_out, 2, idx1)
        v = v0 * (1 - frac[:, None, :]) + v1 * frac[:, None, :]
        return v[:, 0, :], v[:, 1, :].clamp(-6, 3)

    for _ in range(iters):
        g = cnn(_grid_encode(xc, yc, xs_grid, torch))
        mu, log_s = read_off(g, xt)
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
        g = cnn(_grid_encode(xc_t, yc_t, xs_grid, torch))
        mu_t, sig_t = read_off(g, xt_t)
        mu_np, sig_np = mu_t.numpy(), sig_t.exp().numpy()

    mse_cnp = float(np.mean((np.asarray(mu_np) - te[3][..., 0]) ** 2))
    base = _kernel_ridge(te[0], te[1], te[2])
    mse_base = float(np.mean((np.asarray(base) - te[3][..., 0]) ** 2))
    cov90 = float(np.mean(np.abs(te[3][..., 0] - np.asarray(mu_np)) <= 1.645 * np.asarray(sig_np)))
    return {
        "synthetic_convnp_mse": mse_cnp,
        "synthetic_convnp_kernel_ridge_mse": mse_base,
        "synthetic_convnp_mse_gain": mse_base - mse_cnp,
        "synthetic_convnp_cov90": cov90,
        "synthetic_convnp_cov90_err": abs(cov90 - 0.90),
        "synthetic_torch_available": 1.0,
    }
