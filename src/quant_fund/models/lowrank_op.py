"""Low-rank neural operator (Li et al. 2020) — kernel integral operator
approximated by low-rank factors: K f ≈ φ(x)·(Σ_j ψ(x_j) f(x_j)) — vs MLP.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lowrank_op requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lowrank_op(
    seed: int = 713,
    iters: int = 150,
    rank: int = 8,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed)
    at = torch.tensor(a_tr).float()
    ut = torch.tensor(u_tr).float()
    ate = torch.tensor(a_te).float()
    n = a_tr.shape[1]
    torch.manual_seed(seed)
    phi = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, rank))
    psi = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, rank))
    out = torch.nn.Sequential(
        torch.nn.Linear(rank + 1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
    )
    params = list(phi.parameters()) + list(psi.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    xs = torch.linspace(0, 1, n)[:, None].float()

    def fwd(ax):
        # ax: (b, n) input function values at grid
        b = len(ax)
        ph = phi(xs.expand(b, -1, -1))  # (b,n,rank)
        ps = psi(xs.expand(b, -1, -1))
        integ = (ps * ax[:, :, None]).mean(1)  # (b, rank)
        fea = torch.cat([ph * integ[:, None, :], ax[:, :, None]], -1)
        return out(fea).squeeze(-1)

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
        "synthetic_lo_rell2": err,
        "synthetic_lo_mlp_rell2": err_b,
        "synthetic_lo_gain": err_b - err,
        "torch_available": 1.0,
    }
