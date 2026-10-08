"""Graph Neural Operator (Li et al. 2020) — message-passing kernel (SYNTHETIC)
operator on the grid graph: u(x_i) = Σ_j κ(x_i,x_j,a_i,a_j)·f_j — vs MLP.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gno_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gno_lite(
    seed: int = 723,
    iters: int = 120,
    r: int = 6,
    width: int = 24,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed, n_grid=24)
    at = torch.tensor(a_tr).float()
    ut = torch.tensor(u_tr).float()
    ate = torch.tensor(a_te).float()
    n = a_tr.shape[1]
    xs = torch.linspace(0, 1, n)
    # edges: |i-j| <= r
    ei = torch.tensor([[i, j] for i in range(n) for j in range(n) if 0 < abs(i - j) <= r]).T
    torch.manual_seed(seed)
    lift = torch.nn.Linear(2, width)
    kern = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, width * width)
    )
    out = torch.nn.Linear(width, 1)
    params = list(lift.parameters()) + list(kern.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=0.005)

    def fwd(ax):
        b = len(ax)
        v = lift(torch.stack([ax, xs.expand(b, -1)], -1))  # (b,n,w)
        efeat = torch.stack(
            [ax[:, ei[0]], ax[:, ei[1]], xs[ei[0]].expand(b, -1) - xs[ei[1]].expand(b, -1)], -1
        )  # (b,E,3)
        ke = kern(efeat).reshape(b, ei.shape[1], width, width)
        # message: Σ_e κ_e · v_j / deg_i
        msgs = torch.einsum("bewv,bev->bew", ke, v[:, ei[1]])
        agg = torch.zeros(b, n, width)
        deg = torch.zeros(b, n, 1)
        agg.index_add_(1, ei[0], msgs)
        deg.index_add_(1, ei[0], torch.ones(b, ei.shape[1], 1))
        v = torch.nn.functional.gelu(v + agg / deg.clamp_min(1))
        return out(v).squeeze(-1)

    for _i in range(iters):
        loss = ((fwd(at) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = rel_l2(fwd(ate).numpy(), u_te)
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(), torch.nn.Linear(64, n))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((mlp(at) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err_b = rel_l2(mlp(ate).numpy(), u_te)
    return {
        "synthetic_gno_rell2": err,
        "synthetic_gno_mlp_rell2": err_b,
        "synthetic_gno_gain": err_b - err,
        "synthetic_torch_available": 1.0,
    }
