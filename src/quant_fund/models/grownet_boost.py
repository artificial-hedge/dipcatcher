"""GrowNet — gradient boosting by shallow nets (Badirli et al. 2020).

Stage-k net trains on the negative gradient of the ensemble's logit
loss; ensemble = sum of stage outputs — vs single MLP.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("grownet_boost requires torch (pip install -e .[nn])") from exc
    return torch


def bench_grownet_boost(
    seed: int = 601,
    n: int = 400,
    n_stages: int = 4,
    iters: int = 60,
    h: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_t = torch.tensor(x_tr).float()
    y_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    # stage nets accumulate logits
    nets = []
    F = torch.zeros(len(x_t), 2)
    for k in range(n_stages):
        torch.manual_seed(seed + k)
        net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 2))
        # residual targets: y_onehot - softmax(F)
        resid = torch.nn.functional.one_hot(y_t, 2).float() - torch.softmax(F, -1)
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _i in range(iters):
            pred = net(x_t)
            loss = ((pred - resid) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        nets.append(net)
        with torch.no_grad():
            F = F + net(x_t)
    with torch.no_grad():
        F_te = sum(net(x_te_t) for net in nets)
        acc_g = float((F_te.argmax(1) == y_te_t).float().mean())
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _i in range(n_stages * iters):
        loss = torch.nn.functional.cross_entropy(mlp(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_m = acc_of(torch, mlp, x_te_t, y_te_t)
    return {
        "synthetic_gn_acc": acc_g,
        "synthetic_gn_mlp_acc": acc_m,
        "synthetic_gn_gain": acc_g - acc_m,
        "synthetic_torch_available": 1.0,
    }
