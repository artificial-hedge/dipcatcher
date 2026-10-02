"""One-shot / weight-sharing NAS (Bender et al. 2018).

Train a single supernet whose width is masked per-sample; rank
subnets by inherited-weight accuracy. Rank correlation between
supernet ranking and standalone-trained ranking.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import acc_of, split
from quant_fund.models._nas_synth import _torch, build_net, noisy_labels


def bench_one_shot_nas(
    seed: int = 487,
    n: int = 300,
    iters: int = 80,
    hidds=(4, 8, 16, 24, 48),
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    hmax = max(hidds)
    torch.manual_seed(seed)
    supernet = torch.nn.Sequential(
        torch.nn.Linear(8, hmax), torch.nn.ReLU(), torch.nn.Linear(hmax, 2)
    )
    opt = torch.optim.Adam(supernet.parameters(), lr=0.02)
    for _i in range(iters):
        h = hidds[int(np.random.default_rng().integers(len(hidds)))]
        w1 = supernet[0].weight.clone()
        w2 = supernet[2].weight.clone()
        # masked forward: zero out hidden units > h
        hidden = torch.relu(x_tr_t @ w1[:h].T + supernet[0].bias[:h])
        out = hidden @ w2[:, :h].T + supernet[2].bias
        loss = torch.nn.functional.cross_entropy(out, y_tr_t)
        opt.zero_grad()
        loss.backward()
        # zero grads on unused units
        supernet[0].weight.grad[h:] = 0
        supernet[0].bias.grad[h:] = 0
        supernet[2].weight.grad[:, h:] = 0
        opt.step()
    # subnet ranking by inherited weights
    super_rank = []
    for h in hidds:
        with torch.no_grad():
            hidden = torch.relu(x_te_t @ supernet[0].weight[:h].T + supernet[0].bias[:h])
            out = hidden @ supernet[2].weight[:, :h].T + supernet[2].bias
            super_rank.append(float((out.argmax(1) == y_te_t).float().mean()))
    # standalone ranking
    true_rank = []
    for k, h in enumerate(hidds):
        net = build_net(torch, (h, 1, 0), seed=k)
        opt2 = torch.optim.Adam(net.parameters(), lr=0.02)
        for _i in range(iters):
            loss = torch.nn.functional.cross_entropy(net(x_tr_t), y_tr_t)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        true_rank.append(acc_of(torch, net, x_te_t, y_te_t))
    rho = float(
        np.corrcoef(
            np.argsort(np.argsort(super_rank)),
            np.argsort(np.argsort(true_rank)),
        )[0, 1]
    )
    return {
        "synthetic_os_rank_rho": rho,
        "synthetic_os_best_super_h": float(hidds[int(np.argmax(super_rank))]),
        "synthetic_os_best_true_h": float(hidds[int(np.argmax(true_rank))]),
        "synthetic_os_match": float(np.argmax(super_rank) == np.argmax(true_rank)),
        "torch_available": 1.0,
    }
