"""R2D2 (Bertinetto et al. 2019) — meta-learned feature body + closed-form
ridge-regression head per task (no gradient steps needed). Query MSE vs
kernel-ridge on raw x.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("r2d2_meta requires torch (pip install -e .[nn])") from exc
    return torch


def bench_r2d2_meta(seed: int = 871, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    body = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16))
    opt = torch.optim.Adam(body.parameters(), lr=0.005)
    lam = 1e-3
    for _t in range(n_tasks):
        xs, ys, xq, yq = sine_task(rng, K=K)
        Zs = body(torch.tensor(xs).float()[:, None])
        Z = Zs.detach().numpy()
        w_r = np.linalg.solve(Z.T @ Z + lam * np.eye(16), Z.T @ ys)
        Zq = body(torch.tensor(xq).float()[:, None])
        pred = Zq @ torch.tensor(w_r).float()
        # outer loss on query: grads flow through body via Zq
        loss = ((pred - torch.tensor(yq).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mses, mses_raw = [], []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 6000 + i), K=K)
        with torch.no_grad():
            Zs = body(torch.tensor(xs).float()[:, None]).numpy()
            Zq = body(torch.tensor(xq).float()[:, None]).numpy()
        w_r = np.linalg.solve(Zs.T @ Zs + lam * np.eye(16), Zs.T @ ys)
        pred = Zq @ w_r
        mses.append(float(((pred - yq) ** 2).mean()))
        # raw-x ridge
        Xs = np.stack([xs, np.ones(len(xs))], 1)
        w0 = np.linalg.solve(Xs.T @ Xs + lam * np.eye(2), Xs.T @ ys)
        mses_raw.append(float(((np.stack([xq, np.ones(len(xq))], 1) @ w0 - yq) ** 2).mean()))
    return {
        "synthetic_r2d2_query_mse": float(np.mean(mses)),
        "synthetic_r2d2_raw_mse": float(np.mean(mses_raw)),
        "synthetic_r2d2_gain": float(np.mean(mses_raw) - np.mean(mses)),
        "torch_available": 1.0,
    }
