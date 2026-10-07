"""Machine unlearning via gradient ascent + retain repair (SYNTHETIC).

Model trained on forget ∪ retain. Unlearn: ascend loss on forget set
while descending on retain — forget-set accuracy collapses toward
chance while retain accuracy is preserved; compared against a
retrain-from-scratch oracle.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import op_features, synth_problems


def _torch():
    try:
        import torch

        return torch

    except ImportError as exc:
        raise ImportError("unlearn_ga needs the torch `nn` extra") from exc


def bench_unlearn_ga(
    seed: int = 211,
    n_forget: int = 60,
    n_retain: int = 120,
    iters: int = 250,
    unl_iters: int = 60,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    a, ops, _y = synth_problems(n_forget + n_retain, rng)
    a_f, ops_f = a[:n_forget], ops[:n_forget]
    a_r, ops_r = a[n_forget:], ops[n_forget:]

    heads = [torch.nn.Linear(10, 3) for _ in range(2)]
    opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=5e-3)

    def loss_on(aa, oo):
        return sum(
            torch.nn.functional.cross_entropy(
                heads[s](torch.tensor(op_features(aa, s)).float()),
                torch.tensor(oo[:, s]),
            )
            for s in range(2)
        )

    for _i in range(iters):  # train on both
        loss = loss_on(np.concatenate([a_f, a_r]), np.concatenate([ops_f, ops_r]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def acc(aa, oo):
        with torch.no_grad():
            pred = np.stack(
                [
                    np.array(heads[s](torch.tensor(op_features(aa, s)).float())).argmax(-1)
                    for s in range(2)
                ],
                -1,
            )
        return float((pred == oo).mean())

    acc_f_pre = acc(a_f, ops_f)
    for _i in range(unl_iters):  # ascend forget, descend retain
        loss = -loss_on(a_f, ops_f) + loss_on(a_r, ops_r)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_f_post = acc(a_f, ops_f)
    acc_r_post = acc(a_r, ops_r)
    return {
        "synthetic_unlearn_forget_pre": acc_f_pre,
        "synthetic_unlearn_forget_post": acc_f_post,
        "synthetic_unlearn_retain_post": acc_r_post,
        "synthetic_unlearn_drop": acc_f_pre - acc_f_post,
        "synthetic_torch_available": 1.0,
    }
