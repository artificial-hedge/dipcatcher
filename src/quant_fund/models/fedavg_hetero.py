"""FedAvg under client heterogeneity (McMahan 2017 + Li 2020).

IID vs non-IID client shards: non-IID FedAvg diverges relative to
centralized training; FedProx (proximal term μ) recovers part of the
gap — the heterogeneity effect quantified on the synth task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import acc_of, make_mlp, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("fedavg_hetero requires torch (pip install -e .[nn])") from exc
    return torch


def _client_train(torch, w0, x, y, epochs, lr, mu, w_ref):
    torch.manual_seed(0)
    net = make_mlp(torch)
    net.load_state_dict({k: v.clone() for k, v in w0.items()})
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _e in range(epochs):
        loss = torch.nn.functional.cross_entropy(net(x), y)
        if mu > 0:
            prox = sum(((p - w_ref[k]) ** 2).sum() for k, p in net.named_parameters())
            loss = loss + 0.5 * mu * prox
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _fedavg(torch, shards, w0, rounds, epochs, lr, mu, x_te, y_te):
    w = {k: v.clone() for k, v in w0.items()}
    accs = []
    for _r in range(rounds):
        outs = []
        for xs, ys in shards:
            outs.append(_client_train(torch, w, xs, ys, epochs, lr, mu, w))
        for k in w:
            w[k] = torch.stack([o[k] for o in outs]).mean(0)
        net = make_mlp(torch)
        net.load_state_dict(w)
        accs.append(acc_of(torch, net, x_te, y_te))
    return accs


def bench_fedavg_hetero(
    seed: int = 431,
    n: int = 400,
    n_clients: int = 4,
    rounds: int = 12,
    epochs: int = 25,
    lr: float = 0.05,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    torch.manual_seed(seed)
    w0 = make_mlp(torch).state_dict()
    # IID shard: random split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_tr))
    iid = [(x_tr_t[perm[c::n_clients]], y_tr_t[perm[c::n_clients]]) for c in range(n_clients)]
    # non-IID: covariate shift — client c sees x shifted on dim 0,
    # each learning a different boundary; naive averaging degrades
    perm2 = rng.permutation(len(x_tr))
    niid = []
    for c in range(n_clients):
        xc = x_tr_t[perm2[c::n_clients]].clone()
        xc[:, 0] = xc[:, 0] + 1.5 * (c - 1.5)
        niid.append((xc, y_tr_t[perm2[c::n_clients]]))
    acc_iid = _fedavg(torch, iid, w0, rounds, epochs, lr, 0.0, x_te_t, y_te_t)[-1]
    acc_niid = _fedavg(torch, niid, w0, rounds, epochs, lr, 0.0, x_te_t, y_te_t)[-1]
    acc_prox = _fedavg(torch, niid, w0, rounds, epochs, lr, 0.5, x_te_t, y_te_t)[-1]
    return {
        "synthetic_fed_iid_acc": acc_iid,
        "synthetic_fed_niid_acc": acc_niid,
        "synthetic_fed_prox_acc": acc_prox,
        "synthetic_fed_hetero_gap": acc_iid - acc_niid,
        "synthetic_fed_prox_recovery": acc_prox - acc_niid,
        "synthetic_torch_available": 1.0,
    }
