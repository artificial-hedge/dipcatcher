"""Hessian spectrum — top eigenvalue (sharpness) via power iteration on (SYNTHETIC)
Hessian-vector products; trained net vs untrained baseline sharpness.
"""

from __future__ import annotations

from quant_fund.models._td_synth import make_data, train_mlp


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("hessian_eig requires torch (pip install -e .[nn])") from exc
    return torch


def _top_eig(torch, net, Xt, yt, iters: int = 60) -> float:
    params = list(net.parameters())
    n = sum(p.numel() for p in params)
    v = torch.randn(n)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        g = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([x.reshape(-1) for x in g])
        Hv = torch.autograd.grad(g @ v, params, retain_graph=True)
        Hv = torch.cat([x.reshape(-1) for x in Hv]).detach()
        lam = float(v @ Hv)
        v = Hv / (Hv.norm() + 1e-12)
    return abs(lam)


def bench_hessian_eig(seed: int = 2353) -> dict[str, float]:
    torch = _torch()
    X, y, _, _ = make_data(seed)
    net, _ = train_mlp(torch, X, y, iters=400, seed=seed)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()[:, None]
    lam_trained = _top_eig(torch, net, Xt, yt)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 5)
        net0 = torch.nn.Sequential(torch.nn.Linear(8, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    lam_init = _top_eig(torch, net0, Xt, yt)
    if lam_trained <= 0 or lam_init <= 0:
        raise ValueError("non-positive top eigenvalue")
    return {
        "synthetic_hess_top_trained": lam_trained,
        "synthetic_hess_top_init": lam_init,
        "synthetic_hess_curv_gain": lam_trained - lam_init,
        "synthetic_torch_available": 1.0,
    }
