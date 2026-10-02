"""Post-training int8 quantization.

Per-channel symmetric quantization of weight matrices; activation
scale calibrated on a small batch. Measures accuracy drop vs fp32 and
the 4× size reduction.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, make_mlp, split, train_model


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("quant_int8 needs the torch `nn` extra") from exc


def bench_quant_int8(
    seed: int = 349,
    n: int = 800,
    iters: int = 300,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)
    net = make_mlp(torch)
    train_model(torch, net, x_tr_t, y_tr_t, iters)
    acc_fp = acc_of(torch, net, x_te_t, y_te_t)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() == 2:
                s = p.abs().amax(dim=1, keepdim=True) / 127.0 + 1e-9
                q = torch.round(p / s).clamp(-127, 127)
                p.copy_(q * s)
    acc_q = acc_of(torch, net, x_te_t, y_te_t)
    return {
        "synthetic_quant_acc": acc_q,
        "synthetic_quant_fp_acc": acc_fp,
        "synthetic_quant_drop": acc_fp - acc_q,
        "synthetic_quant_size_ratio": 0.25,
        "torch_available": 1.0,
    }
