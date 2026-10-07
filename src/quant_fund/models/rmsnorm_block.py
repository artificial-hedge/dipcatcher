"""RMSNorm block (Zhang & Sennrich 2019) — RMS-normalized MLP vs
LayerNorm and unnormalized MLP on regime task; measures acc AND
final-layer activation drift (RMS deviation from 1).
"""

from __future__ import annotations

from quant_fund.models._lm_synth import regime_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("rmsnorm_block requires torch (pip install -e .[nn])") from exc
    return torch


def _train(seed: int, mode: str, iters: int = 600) -> tuple[float, float]:
    torch = _torch()
    X, y = regime_task(seed)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    torch.manual_seed(seed)
    D = 32
    W1 = torch.nn.Linear(8, D)
    W2 = torch.nn.Linear(D, D)
    W3 = torch.nn.Linear(D, 1)
    ln = torch.nn.LayerNorm(D)
    rms_w = torch.nn.Parameter(torch.ones(D))
    params = (
        list(W1.parameters())
        + list(W2.parameters())
        + list(W3.parameters())
        + list(ln.parameters())
        + [rms_w]
    )

    def fwd(x):
        h = torch.relu(W1(x))
        if mode == "ln":
            h = ln(h)
        elif mode == "rms":
            h = h * rms_w / h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        h = torch.relu(W2(h))
        if mode == "ln":
            h = ln(h)
        elif mode == "rms":
            h = h * rms_w / h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return W3(h).squeeze(-1), h

    opt = torch.optim.Adam(params, lr=0.02)
    for _ in range(iters):
        out, _ = fwd(Xt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    X2, y2 = regime_task(seed + 1)
    Xt2 = torch.tensor(X2).float()
    out, h = fwd(Xt2)
    acc = float(((out > 0).float().numpy() == y2).mean())
    drift = float((h.pow(2).mean(-1).sqrt() - 1).abs().mean())
    return acc, drift


def bench_rmsnorm_block(seed: int = 1719, iters: int = 600) -> dict[str, float]:
    acc_r, dr_r = _train(seed, "rms", iters)
    acc_l, dr_l = _train(seed + 1, "ln", iters)
    acc_n, dr_n = _train(seed + 2, "none", iters)
    return {
        "synthetic_rms_acc": acc_r,
        "synthetic_ln_acc": acc_l,
        "synthetic_nonorm_acc": acc_n,
        "synthetic_rms_drift": dr_r,
        "synthetic_nonorm_drift": dr_n,
        "synthetic_rms_gain": acc_r - acc_n,
        "synthetic_torch_available": 1.0,
    }
