"""Fisher-information structured pruning (Theis et al.).

Score each hidden unit by E[(∂L/∂w)²]-weighted magnitude (Fisher);
prune the lowest-score units. Fisher pruning retains accuracy better
than random unit drop at the same sparsity.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, make_mlp, split, train_model


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("fisher_prune needs the torch `nn` extra") from exc


def bench_fisher_prune(
    seed: int = 367,
    n: int = 800,
    drop_frac: float = 0.5,
    iters: int = 300,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)
    net = make_mlp(torch, 24)
    train_model(torch, net, x_tr_t, y_tr_t, iters)
    acc_full = acc_of(torch, net, x_te_t, y_te_t)
    # Fisher score per hidden unit: mean grad² on w1 columns + w2 rows
    loss = torch.nn.functional.cross_entropy(net(x_tr_t), y_tr_t)
    g = torch.autograd.grad(loss, list(net.parameters()))
    w1_g, w2_g = g[0], g[2]
    score = (w1_g.abs() * net[0].weight.abs()).mean(1) + (w2_g.abs() * net[2].weight.abs()).sum(0)
    keep = score.argsort(descending=True)[: int(24 * (1 - drop_frac))]
    mask = torch.zeros(24)
    mask[keep] = 1.0
    with torch.no_grad():
        net[0].weight.mul_(mask[:, None])
        net[2].weight.mul_(mask[None, :])
    acc_f = acc_of(torch, net, x_te_t, y_te_t)
    # random unit mask
    torch.manual_seed(seed + 2)
    net2 = make_mlp(torch, 24)
    train_model(torch, net2, x_tr_t, y_tr_t, iters)
    rkeep = torch.randperm(24)[: int(24 * (1 - drop_frac))]
    rmask = torch.zeros(24)
    rmask[rkeep] = 1.0
    with torch.no_grad():
        net2[0].weight.mul_(rmask[:, None])
        net2[2].weight.mul_(rmask[None, :])
    acc_r = acc_of(torch, net2, x_te_t, y_te_t)
    return {
        "synthetic_fisher_acc": acc_f,
        "synthetic_fisher_full_acc": acc_full,
        "synthetic_fisher_random_acc": acc_r,
        "synthetic_fisher_gain": acc_f - acc_r,
        "torch_available": 1.0,
    }
