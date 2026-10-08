"""MOON (Li et al. 2021) — model-contrastive: local loss includes a (SYNTHETIC)
contrastive term pulling local representations toward the global model's
and away from the previous local round — vs FedAvg under label skew.
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("moon_fl requires torch (pip install -e .[nn])") from exc
    return torch


def bench_moon_fl(
    seed: int = 701,
    n_clients: int = 4,
    rounds: int = 8,
    local_ep: int = 10,
    n: int = 60,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    # non-IID split: sort by label, partition contiguous blocks → client i
    # sees mostly one class (the FedAvg pathology MOON targets)
    x_all = rng.standard_normal((n * n_clients, 8))
    y_all = (x_all[:, 0] + 0.5 * x_all[:, 1] > 0).astype(float)
    order = np.argsort(y_all)
    x_all, y_all = x_all[order], y_all[order]
    shards = [
        (
            torch.tensor(x_all[i * n : (i + 1) * n]).float(),
            torch.tensor(y_all[i * n : (i + 1) * n]).long(),
        )
        for i in range(n_clients)
    ]
    x_te = rng.standard_normal((200, 8))
    y_te = (x_te[:, 0] + 0.5 * x_te[:, 1] > 0).astype(float)
    xt = torch.tensor(x_te).float()
    yt = torch.tensor(y_te).long()

    def make():
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))

    def acc(net):
        with torch.no_grad():
            return float((net(xt).argmax(1) == yt).float().mean())

    def fedavg(moon: bool, mu: float = 1.0):
        glob = make()
        prev_locals = [make() for _ in shards]
        for _r in range(rounds):
            gstate = {k: v.clone() for k, v in glob.state_dict().items()}
            locals_ = []
            for i, (xc, yc) in enumerate(shards):
                loc = make()
                loc.load_state_dict(gstate)
                opt = torch.optim.SGD(loc.parameters(), lr=0.1)
                for _e in range(local_ep):
                    logits = loc(xc)
                    loss = torch.nn.functional.cross_entropy(logits, yc)
                    if moon:
                        with torch.no_grad():
                            z_g = torch.nn.functional.normalize(
                                torch.cat([p.flatten() for p in glob.parameters()]), dim=0
                            )
                            z_p = torch.nn.functional.normalize(
                                torch.cat([p.flatten() for p in prev_locals[i].parameters()]), dim=0
                            )
                        z_l = torch.nn.functional.normalize(
                            torch.cat([p.flatten() for p in loc.parameters()]), dim=0
                        )
                        loss = loss + mu * (
                            torch.nn.functional.logsigmoid(-((z_l - z_g) ** 2).sum())
                            + torch.nn.functional.logsigmoid(((z_l - z_p) ** 2).sum())
                        )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                prev_locals[i] = loc
                locals_.append(loc)
            with torch.no_grad():
                sd = glob.state_dict()
                for k in sd:
                    sd[k] = (
                        torch.stack([lc.state_dict()[k].float() for lc in locals_])
                        .mean(0)
                        .to(sd[k].dtype)
                    )
                glob.load_state_dict(sd)
        return acc(glob)

    a_moon = fedavg(moon=True)
    a_fa = fedavg(moon=False)
    return {
        "synthetic_moon_acc": a_moon,
        "synthetic_moon_fedavg_acc": a_fa,
        "synthetic_moon_gain": a_moon - a_fa,
        "synthetic_torch_available": 1.0,
    }
