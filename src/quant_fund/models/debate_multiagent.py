"""Multi-agent debate (Du et al. 2023) (SYNTHETIC).

Three op-policies trained on disjoint noisy-label subsets disagree;
debate = averaging logits + argmax (consensus) beats the median
single agent — the ensemble-of-reasoners effect.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import eval_chain, op_features, synth_problems


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("debate_multiagent needs the torch `nn` extra") from exc


def bench_debate_multiagent(
    seed: int = 199,
    n_train: int = 90,
    n_test: int = 300,
    label_noise: float = 0.25,
    iters: int = 250,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    a, ops, y = synth_problems(n_train + n_test, rng)
    a_tr, ops_tr = a[:n_train], ops[:n_train]
    a_te, _ops_te, y_te = a[n_train:], ops[n_train:], y[n_train:]

    n_agents = 3
    per = n_train // n_agents
    agents = [[torch.nn.Linear(10, 3) for _ in range(2)] for _ in range(n_agents)]
    for g, heads in enumerate(agents):
        idx = slice(g * per, (g + 1) * per)
        noisy = ops_tr[idx].copy()
        flip = rng.random(noisy.shape) < label_noise
        noisy[flip] = rng.integers(0, 3, flip.sum())
        opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=5e-3)
        for _i in range(iters):
            loss = sum(
                torch.nn.functional.cross_entropy(
                    heads[s](torch.tensor(op_features(a_tr[idx], s)).float()),
                    torch.tensor(noisy[:, s]),
                )
                for s in range(2)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = np.zeros((n_agents, 2, n_test, 3))
        for g, heads in enumerate(agents):
            for s in range(2):
                logits[g, s] = np.array(heads[s](torch.tensor(op_features(a_te, s)).float()))
        single = np.zeros(n_agents)
        for g in range(n_agents):
            pred = np.stack([logits[g, 0].argmax(-1), logits[g, 1].argmax(-1)], -1)
            single[g] = float((eval_chain(a_te, pred) == y_te).mean())
        debate = np.stack([logits[:, 0].mean(0).argmax(-1), logits[:, 1].mean(0).argmax(-1)], -1)
        acc_debate = float((eval_chain(a_te, debate) == y_te).mean())
    return {
        "synthetic_debate_acc": acc_debate,
        "synthetic_debate_single_mean": float(single.mean()),
        "synthetic_debate_single_max": float(single.max()),
        "synthetic_debate_gain": acc_debate - float(single.max()),
        "synthetic_torch_available": 1.0,
    }
