"""SwiGLU FFN (Shazeer 2020) — SwiGLU gate FFN vs plain ReLU MLP on
the multi-regime task: sigmoid(x @ w_r) regime-dependent labels.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import regime_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("swiglu_ffn requires torch (pip install -e .[nn])") from exc
    return torch


def _train(seed: int, swiglu: bool, iters: int = 600) -> float:
    torch = _torch()
    X, y = regime_task(seed)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    torch.manual_seed(seed)
    D = 32
    if swiglu:
        W1 = torch.nn.Linear(8, 2 * D)
        W2 = torch.nn.Linear(D, 1)
        params = list(W1.parameters()) + list(W2.parameters())

        def fwd(x):
            h = W1(x)
            return W2(torch.nn.functional.silu(h[:, :D]) * h[:, D:]).squeeze(-1)
    else:
        W1 = torch.nn.Linear(8, D)
        W2 = torch.nn.Linear(D, 1)
        params = list(W1.parameters()) + list(W2.parameters())

        def fwd(x):
            return W2(torch.relu(W1(x))).squeeze(-1)

    opt = torch.optim.Adam(params, lr=0.01)
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(fwd(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    X2, y2 = regime_task(seed + 1)
    Xt2 = torch.tensor(X2).float()
    return float(((fwd(Xt2) > 0).float().numpy() == y2).mean())


def bench_swiglu_ffn(seed: int = 1713, iters: int = 600) -> dict[str, float]:
    acc_s = _train(seed, True, iters)
    acc_r = _train(seed + 1, False, iters)
    return {
        "synthetic_swiglu_acc": acc_s,
        "synthetic_relu_acc": acc_r,
        "synthetic_swiglu_gain": acc_s - acc_r,
        "synthetic_torch_available": 1.0,
    }
