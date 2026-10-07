"""Iterative magnitude pruning (Han et al. 2015).

Globally prune the smallest-magnitude weights to sparsity s, then
fine-tune. Accuracy-at-sparsity vs a random-mask control shows the
magnitude heuristic actually finds the important connections.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, make_mlp, split, train_model


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("magnitude_pruning needs the torch `nn` extra") from exc


def bench_magnitude_pruning(
    seed: int = 337,
    n: int = 800,
    sparsity: float = 0.7,
    iters: int = 300,
    ft_iters: int = 100,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)

    net = make_mlp(torch)
    train_model(torch, net, x_tr_t, y_tr_t, iters)
    acc_dense = acc_of(torch, net, x_te_t, y_te_t)
    # global magnitude mask
    params = [p for p in net.parameters() if p.dim() == 2]
    allw = torch.cat([p.detach().abs().flatten() for p in params])
    thresh = allw.kthvalue(int(sparsity * len(allw))).values
    masks = [(p.detach().abs() >= thresh).float() for p in params]
    full_masks = []
    pi = 0
    for p in net.parameters():
        if p.dim() == 2:
            full_masks.append(masks[pi])
            pi += 1
        else:
            full_masks.append(torch.ones_like(p))
    with torch.no_grad():
        for p, m in zip(net.parameters(), full_masks, strict=True):
            p *= m
    train_model(torch, net, x_tr_t, y_tr_t, ft_iters, mask=full_masks)
    acc_pruned = acc_of(torch, net, x_te_t, y_te_t)
    # random mask control
    net2 = make_mlp(torch)
    train_model(torch, net2, x_tr_t, y_tr_t, iters)
    rmasks = []
    for p in net2.parameters():
        if p.dim() == 2:
            rmasks.append((torch.rand_like(p) > sparsity).float())
        else:
            rmasks.append(torch.ones_like(p))
    with torch.no_grad():
        for p, m in zip(net2.parameters(), rmasks, strict=True):
            p *= m
    train_model(torch, net2, x_tr_t, y_tr_t, ft_iters, mask=rmasks)
    acc_rand = acc_of(torch, net2, x_te_t, y_te_t)
    return {
        "synthetic_mprune_acc": acc_pruned,
        "synthetic_mprune_dense_acc": acc_dense,
        "synthetic_mprune_random_acc": acc_rand,
        "synthetic_mprune_gain": acc_pruned - acc_rand,
        "synthetic_mprune_retention": acc_pruned / max(acc_dense, 1e-9),
        "synthetic_torch_available": 1.0,
    }
