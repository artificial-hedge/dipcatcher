"""Reptile (Nichol et al. 2018) — first-order meta-init: for each task,
train a few SGD steps from the meta-init, then move the meta-init toward
the task solution (w ← w + ε·(w_t − w)). Query MSE after 5-shot vs pooled.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _net():
    import torch

    def make():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    return torch, make


def _train(net, xs, ys, iters: int = 30, lr: float = 0.01):
    import torch

    opt = torch.optim.SGD(net.parameters(), lr=lr)
    X = torch.tensor(xs).float()[:, None]
    Y = torch.tensor(ys).float()[:, None]
    for _ in range(iters):
        loss = ((net(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def _mse(net, xq, yq) -> float:
    import torch

    with torch.no_grad():
        return float(
            (
                (net(torch.tensor(xq).float()[:, None]).squeeze(-1) - torch.tensor(yq).float()) ** 2
            ).mean()
        )


def bench_reptile(seed: int = 853, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch, Net = _net()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    meta = Net()
    eps = 0.5
    for _t in range(n_tasks):
        xs, ys, _xq, _yq = sine_task(rng, K=K)
        worker = Net()
        worker.load_state_dict(meta.state_dict())
        _train(worker, xs, ys, iters=25)
        with torch.no_grad():
            for pm, pw in zip(meta.parameters(), worker.parameters(), strict=True):
                pm.add_(eps * (pw - pm))
    # eval: adapt 25 steps on support of a fresh task
    mses, mses_pooled = [], []
    pooled = Net()  # non-meta pooled net on all train data
    xs_all = rng.uniform(-5, 5, 200)
    A_all = rng.uniform(0.1, 5, 200)
    phi_all = rng.uniform(0, np.pi, 200)
    _train(
        pooled,
        xs_all,
        np.asarray([a * np.sin(x + p) for a, x, p in zip(A_all, xs_all, phi_all, strict=True)]),
        iters=60,
    )
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 1000 + i), K=K)
        w1 = Net()
        w1.load_state_dict(meta.state_dict())
        _train(w1, xs, ys, iters=25)
        mses.append(_mse(w1, xq, yq))
        w2 = Net()
        w2.load_state_dict(pooled.state_dict())
        _train(w2, xs, ys, iters=25)
        mses_pooled.append(_mse(w2, xq, yq))
    return {
        "synthetic_rep_query_mse": float(np.mean(mses)),
        "synthetic_rep_pooled_mse": float(np.mean(mses_pooled)),
        "synthetic_rep_gain": float(np.mean(mses_pooled) - np.mean(mses)),
        "torch_available": 1.0,
    }
