"""Low-rank weight factorization (SVD compression) (SYNTHETIC).

Truncate the first-layer weight SVD to rank r → (8×r + r×24) params
vs 8×24; accuracy retention vs parameter fraction.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, make_mlp, split, train_model


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("lowrank_factor needs the torch `nn` extra") from exc


def bench_lowrank_factor(
    seed: int = 359,
    n: int = 800,
    rank: int = 4,
    iters: int = 300,
    ft_iters: int = 100,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)
    net = make_mlp(torch, 24)
    train_model(torch, net, x_tr_t, y_tr_t, iters)
    acc_full = acc_of(torch, net, x_te_t, y_te_t)
    w = net[0].weight.detach()
    U, S, V = torch.linalg.svd(w, full_matrices=False)
    net[0].weight.data = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank]
    train_model(torch, net, x_tr_t, y_tr_t, ft_iters)
    acc_lr = acc_of(torch, net, x_te_t, y_te_t)
    frac = (8 * rank + rank * 24) / (8 * 24)
    return {
        "synthetic_lr_acc": acc_lr,
        "synthetic_lr_full_acc": acc_full,
        "synthetic_lr_retention": acc_lr / max(acc_full, 1e-9),
        "synthetic_lr_param_frac": float(frac),
        "synthetic_torch_available": 1.0,
    }
