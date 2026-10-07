"""TabM-lite — parameter-efficient multi-head ensemble (Gorishniy 2024).

K prediction heads share one trunk (multi-bet ensembling); mean-head
accuracy vs single-head MLP at matched compute.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("tabm_mini requires torch (pip install -e .[nn])") from exc
    return torch


def bench_tabm_mini(
    seed: int = 613,
    n: int = 400,
    iters: int = 120,
    k: int = 8,
    h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_t = torch.tensor(x_tr).float()
    y_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    torch.manual_seed(seed)
    trunk = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU())
    heads = torch.nn.Parameter(torch.randn(k, h, 2) * 0.1)
    bhead = torch.nn.Parameter(torch.zeros(k, 2))
    opt = torch.optim.Adam(list(trunk.parameters()) + [heads, bhead], lr=0.01)
    for _i in range(iters):
        z = trunk(x_t)  # (n,h)
        logits = torch.einsum("nh,khc->nkc", z, heads) + bhead
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y_t.repeat_interleave(k))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = trunk(x_te_t)
        logits = torch.einsum("nh,khc->nkc", z, heads) + bhead
        acc_tabm = float((logits.mean(1).argmax(1) == y_te_t).float().mean())
        acc_best_head = max(
            float((logits[:, kk].argmax(1) == y_te_t).float().mean()) for kk in range(k)
        )
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(mlp(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_m = acc_of(torch, mlp, x_te_t, y_te_t)
    return {
        "synthetic_tabm_acc": acc_tabm,
        "synthetic_tabm_best_head": acc_best_head,
        "synthetic_tabm_mlp_acc": acc_m,
        "synthetic_tabm_gain": acc_tabm - acc_m,
        "synthetic_torch_available": 1.0,
    }
