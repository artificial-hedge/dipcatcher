"""TCN forecaster: dilated causal convolutions over time.

Bai, Kolter & Koltun 2018: causal convs with exponentially dilated
kernels give a receptive field that grows exponentially in depth while
preserving causality — cheaper than attention for long horizons.

Bench: synthetic series with long-lag nonlinear dependence (event at
t-24 modulates t); TCN's dilated field should beat a 4-lag MLP and
an AR(4) baseline.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tcn_forecaster requires the `nn` extra (make sync)") from exc


def synth_long_lag(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """(n, win) inputs; target = nonlinear fn of x[t-24] * x[t-1]."""
    x = np.zeros((n, win))
    y = np.zeros(n)
    for i in range(n):
        z = np.cumsum(0.2 * rng.standard_normal(win))
        x[i] = z - z.mean()
        y[i] = 1.0 * np.tanh(0.9 * z[-24]) + 0.2 * z[-1] + 0.15 * rng.standard_normal()
    return x.astype(np.float64), y.astype(np.float64)


def _tcn_block(torch: Any, ch_in: int, ch_out: int, kernel: int, dilation: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Conv1d(ch_in, ch_out, kernel, padding=dilation * (kernel - 1), dilation=dilation),
        torch.nn.ReLU(),
        torch.nn.Conv1d(ch_out, ch_out, kernel, padding=dilation * (kernel - 1), dilation=dilation),
        torch.nn.ReLU(),
    )


def bench_tcn_forecaster(
    seed: int = 20261231,
    n: int = 320,
    win: int = 48,
    iters: int = 700,
) -> dict[str, float]:
    """Dilated-TCN next-step forecast vs short-window MLP and AR(4)."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    xs, y = synth_long_lag(n, win, rng)
    xs = (xs - xs.mean()) / (xs.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    xb = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)  # (B,1,T)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)

    blocks = torch.nn.ModuleList(
        [_tcn_block(torch, 1 if i == 0 else 16, 16, 3, 2**i) for i in range(4)]
    )
    head = torch.nn.Linear(16, 1)
    params = torch.nn.ModuleList([blocks, head])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _ in range(iters):
        h = xb[tr]
        for b in blocks:
            h = b(h)[:, :, -xb[tr].shape[2] :]
        pred = head(h[:, :, -1])
        loss = torch.mean((pred.squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = xb[te]
        for b in blocks:
            h = b(h)[:, :, -xb[te].shape[2] :]
        tcn_mae = float(torch.mean(torch.abs(head(h[:, :, -1]).squeeze(1) - yb[te].squeeze(1))))

    # short-window MLP: sees only last 4 lags — cannot reach t-24
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    pm = torch.nn.ModuleList([mlp])
    optm = torch.optim.Adam(pm.parameters(), lr=2e-3)
    xshort = xb[tr, 0, -4:]
    for _ in range(iters):
        loss = torch.mean((mlp(xshort).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optm.zero_grad()
        loss.backward()
        optm.step()
    with torch.no_grad():
        mlp_mae = float(torch.mean(torch.abs(mlp(xb[te, 0, -4:]).squeeze(1) - yb[te].squeeze(1))))

    X4 = np.stack([xs[tr, -1 - k] for k in range(4)], 1)
    beta = np.asarray(np.linalg.solve(X4.T @ X4 + 1e-3 * np.eye(4), X4.T @ y[tr]))
    ar_mae = float(np.mean(np.abs(np.stack([xs[te, -1 - k] for k in range(4)], 1) @ beta - y[te])))
    return {
        "synthetic_tcn_mae": tcn_mae,
        "synthetic_tcn_shortmlp_mae": mlp_mae,
        "synthetic_tcn_ar_mae": ar_mae,
        "synthetic_tcn_margin_vs_ar": ar_mae - tcn_mae,
        "synthetic_tcn_margin_vs_shortmlp": mlp_mae - tcn_mae,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_tcn_forecaster()))
