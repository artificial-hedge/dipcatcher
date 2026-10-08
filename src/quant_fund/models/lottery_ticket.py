"""Lottery ticket hypothesis (Frankle & Carbin 2019) (SYNTHETIC).

Rewind the surviving weights to init inside the magnitude mask and
retrain — the "winning ticket" matches dense accuracy while a mask on
re-randomized weights does not.
"""

from __future__ import annotations

import copy

from quant_fund.models._compress_synth import acc_of, make_mlp, split, train_model


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("lottery_ticket needs the torch `nn` extra") from exc


def bench_lottery_ticket(
    seed: int = 347,
    n: int = 800,
    sparsity: float = 0.6,
    iters: int = 300,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)

    net = make_mlp(torch)
    init = copy.deepcopy(net.state_dict())
    train_model(torch, net, x_tr_t, y_tr_t, iters)
    acc_dense = acc_of(torch, net, x_te_t, y_te_t)
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
    # winning ticket: rewind to init, then apply mask
    net.load_state_dict(init)
    with torch.no_grad():
        for p, m in zip(net.parameters(), full_masks, strict=True):
            p *= m
    train_model(torch, net, x_tr_t, y_tr_t, iters, mask=full_masks)
    acc_ticket = acc_of(torch, net, x_te_t, y_te_t)
    # random-init control inside same mask
    torch.manual_seed(seed + 1)
    net3 = make_mlp(torch)
    with torch.no_grad():
        for p, m in zip(net3.parameters(), full_masks, strict=True):
            p *= m
    train_model(torch, net3, x_tr_t, y_tr_t, iters, mask=full_masks)
    acc_rand = acc_of(torch, net3, x_te_t, y_te_t)
    return {
        "synthetic_lth_ticket_acc": acc_ticket,
        "synthetic_lth_dense_acc": acc_dense,
        "synthetic_lth_random_acc": acc_rand,
        "synthetic_lth_gain": acc_ticket - acc_rand,
        "synthetic_torch_available": 1.0,
    }
