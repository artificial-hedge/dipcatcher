"""PINO-lite (Li et al. 2021) — FNO trained with a physics residual: (SYNTHETIC)
data loss + λ‖-∂²û − a‖ where ∂² uses central differences — the
physics penalty should cut error at small data budgets vs data-only.
"""

from __future__ import annotations

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("pino_residual requires torch (pip install -e .[nn])") from exc
    return torch


def _mk(width: int, modes: int):
    import torch

    lift = torch.nn.Conv1d(1, width, 1)
    spec_w = torch.nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * 0.05)
    pw = torch.nn.Conv1d(width, width, 1)
    out = torch.nn.Conv1d(width, 1, 1)
    return lift, spec_w, pw, out


def bench_pino_residual(
    seed: int = 719,
    iters: int = 120,
    modes: int = 8,
    width: int = 24,
    lam: float = 0.02,
    n_keep: int = 40,
) -> dict[str, float]:
    torch = _torch()
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=seed)
    # small-data regime: keep n_keep training pairs
    at = torch.tensor(a_tr[:n_keep]).float()[:, None, :]
    ut = torch.tensor(u_tr[:n_keep]).float()[:, None, :]
    ate = torch.tensor(a_te).float()[:, None, :]
    n = a_tr.shape[1]
    dx = 1.0 / (n - 1)

    def train(physics: bool):
        torch.manual_seed(seed)
        lift, spec_w, pw, out = _mk(width, modes)
        params = list(lift.parameters()) + list(pw.parameters()) + list(out.parameters()) + [spec_w]
        opt = torch.optim.Adam(params, lr=0.01)

        def fwd(ax):
            z = lift(ax)
            zf = torch.fft.rfft(z, dim=-1)
            kept = torch.einsum("bim,iom->bom", zf[:, :, :modes], spec_w)
            zf2 = torch.zeros_like(zf)
            zf2[:, :, :modes] = kept
            z = torch.fft.irfft(zf2, n=n, dim=-1) + pw(z)
            return out(torch.nn.functional.gelu(z))

        for _i in range(iters):
            pred = fwd(at)
            loss = ((pred - ut) ** 2).mean()
            if physics:
                d2u = (pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]) / dx**2
                a_mid = at[:, :, 1:-1]
                loss = loss + lam * ((-d2u - a_mid) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return fwd(ate)[:, 0].numpy()

    pred_p = train(True)
    pred_d = train(False)
    err_p = rel_l2(pred_p, u_te)
    err_d = rel_l2(pred_d, u_te)
    return {
        "synthetic_pino_rell2": err_p,
        "synthetic_pino_data_rell2": err_d,
        "synthetic_pino_gain": err_d - err_p,
        "synthetic_torch_available": 1.0,
    }
