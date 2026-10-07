"""DeepONet (Lu et al. 2021) — branch encodes the input function's (SYNTHETIC)
values, trunk encodes query locations; u(y) = <branch(a), trunk(y)>.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deeponet requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deeponet(
    seed: int = 711,
    iters: int = 150,
    p: int = 32,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed)
    at = torch.tensor(a_tr).float()
    ut = torch.tensor(u_tr).float()
    ate = torch.tensor(a_te).float()
    n_grid = a_tr.shape[1]
    xs = torch.linspace(0, 1, n_grid)[None, :, None].expand(len(at), -1, -1).reshape(-1, 1)
    xs_te = torch.linspace(0, 1, n_grid)[None, :, None].expand(len(ate), -1, -1).reshape(-1, 1)
    torch.manual_seed(seed)
    branch = torch.nn.Sequential(
        torch.nn.Linear(n_grid, 64), torch.nn.ReLU(), torch.nn.Linear(64, p)
    )
    trunk = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, p))
    params = list(branch.parameters()) + list(trunk.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    for _i in range(iters):
        b = branch(at)  # (n,p)
        t = trunk(xs).reshape(len(at), n_grid, p)
        pred = (b[:, None, :] * t).sum(-1)
        loss = ((pred - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        b = branch(ate)
        t = trunk(xs_te).reshape(len(ate), n_grid, p)
        pred = (b[:, None, :] * t).sum(-1).numpy()
    err = rel_l2(pred, u_te)
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(n_grid, 64), torch.nn.ReLU(), torch.nn.Linear(64, n_grid)
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((mlp(at) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err_b = rel_l2(mlp(ate).numpy(), u_te)
    return {
        "synthetic_don_rell2": err,
        "synthetic_don_mlp_rell2": err_b,
        "synthetic_don_gain": err_b - err,
        "synthetic_torch_available": 1.0,
    }
