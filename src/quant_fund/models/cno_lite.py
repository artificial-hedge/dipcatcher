"""Convolutional Neural Operator (Raonic et al. 2024) — U-Net-shaped (SYNTHETIC)
operator: channel lifting + down/up blocks preserve band structure —
vs plain MLP on the Poisson fixture.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cno_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_cno_lite(
    seed: int = 727,
    iters: int = 150,
    width: int = 24,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed)
    at = torch.tensor(a_tr).float()[:, None, :]
    ut = torch.tensor(u_tr).float()[:, None, :]
    ate = torch.tensor(a_te).float()[:, None, :]
    n = a_tr.shape[1]
    torch.manual_seed(seed)
    enc1 = torch.nn.Conv1d(1, width, 3, padding=1)
    down = torch.nn.Conv1d(width, width, 4, stride=2, padding=1)
    mid = torch.nn.Conv1d(width, width, 3, padding=1)
    up = torch.nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
    dec = torch.nn.Conv1d(width, 1, 3, padding=1)
    params = (
        list(enc1.parameters())
        + list(down.parameters())
        + list(mid.parameters())
        + list(up.parameters())
        + list(dec.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.01)
    act = torch.nn.functional.gelu

    def fwd(ax):
        z = act(enc1(ax))
        zd = act(down(z))
        zd = act(mid(zd))
        zu = act(up(zd))[:, :, :n]
        return dec(zu + z)

    for _i in range(iters):
        loss = ((fwd(at) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = rel_l2(fwd(ate)[:, 0].numpy(), u_te)
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(), torch.nn.Linear(64, n))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _i in range(iters):
        loss = ((mlp(at[:, 0]) - ut[:, 0]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err_b = rel_l2(mlp(ate[:, 0]).numpy(), u_te)
    return {
        "synthetic_cno_rell2": err,
        "synthetic_cno_mlp_rell2": err_b,
        "synthetic_cno_gain": err_b - err,
        "synthetic_torch_available": 1.0,
    }
