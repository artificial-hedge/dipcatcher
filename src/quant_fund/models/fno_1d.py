"""Fourier Neural Operator (Li et al. 2021) — spectral convolution:
RFT → keep low modes → learned complex weights → iRFT, lifted through a
pointwise net — vs plain CNN/MLP operator baseline.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("fno_1d requires torch (pip install -e .[nn])") from exc
    return torch


def bench_fno_1d(
    seed: int = 709,
    iters: int = 150,
    modes: int = 8,
    width: int = 24,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed)
    at = torch.tensor(a_tr).float()[:, None, :]
    ut = torch.tensor(u_tr).float()[:, None, :]
    ate = torch.tensor(a_te).float()[:, None, :]
    torch.manual_seed(seed)
    lift = torch.nn.Conv1d(1, width, 1)
    spec_w = torch.nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * 0.05)
    pw = torch.nn.Conv1d(width, width, 1)
    out = torch.nn.Conv1d(width, 1, 1)
    params = list(lift.parameters()) + list(pw.parameters()) + list(out.parameters()) + [spec_w]
    opt = torch.optim.Adam(params, lr=0.01)
    n = a_tr.shape[1]

    def fwd(ax):
        z = lift(ax)
        zf = torch.fft.rfft(z, dim=-1)
        kept = torch.einsum("bim,iom->bom", zf[:, :, :modes], spec_w)
        zf2 = torch.zeros_like(zf)
        zf2[:, :, :modes] = kept
        z = torch.fft.irfft(zf2, n=n, dim=-1) + pw(z)
        return out(torch.nn.functional.gelu(z))

    for _i in range(iters):
        loss = ((fwd(at) - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = fwd(ate)[:, 0].numpy()
    err = rel_l2(pred, u_te)
    # MLP baseline
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
        "synthetic_fno_rell2": err,
        "synthetic_fno_mlp_rell2": err_b,
        "synthetic_fno_gain": err_b - err,
        "synthetic_torch_available": 1.0,
    }
