"""Self-consistency (Wang et al. 2023) — sample-and-vote decoding.

A noisy op-policy sampled K times at temperature τ produces diverse
op chains; majority vote over final values beats the greedy single
sample — the canonical cheap test-time-compute win.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import eval_chain, op_features, synth_problems


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("consistency_vote needs the torch `nn` extra") from exc


def bench_consistency_vote(
    seed: int = 191,
    n_train: int = 24,
    n_test: int = 300,
    k: int = 16,
    tau: float = 0.5,
    iters: int = 200,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    a, ops, y = synth_problems(n_train + n_test, rng)
    a_tr, ops_tr = a[:n_train], ops[:n_train]
    a_te, _ops_te, y_te = a[n_train:], ops[n_train:], y[n_train:]

    heads = [torch.nn.Linear(10, 3) for _ in range(2)]
    opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=5e-3)
    for _i in range(iters):
        loss = sum(
            torch.nn.functional.cross_entropy(
                heads[s](torch.tensor(op_features(a_tr, s)).float()),
                torch.tensor(ops_tr[:, s]),
            )
            for s in range(2)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict(logits_fn):
        sampled = np.zeros((n_test, k, 2), dtype=np.int64)
        greedy = np.zeros((n_test, 2), dtype=np.int64)
        for s in range(2):
            logits = logits_fn(s)
            greedy[:, s] = logits.argmax(-1)
            probs = torch.softmax(logits / tau, -1)
            sampled[:, :, s] = np.array(
                [np.array(torch.multinomial(probs[i], k, replacement=True)) for i in range(n_test)]
            )
        return greedy, sampled

    with torch.no_grad():
        greedy, sampled = predict(lambda s: heads[s](torch.tensor(op_features(a_te, s)).float()))

    acc_greedy = float((eval_chain(a_te, greedy) == y_te).mean())
    votes = np.zeros((n_test, k))
    for j in range(k):
        votes[:, j] = eval_chain(a_te, sampled[:, j, :])
    voted = np.zeros(n_test)
    for i in range(n_test):
        vals, cnt = np.unique(votes[i], return_counts=True)
        voted[i] = vals[np.argmax(cnt)]
    acc_sc = float((voted == y_te).mean())
    acc_sample = float((votes == y_te[:, None]).mean())
    return {
        "synthetic_sc_acc": acc_sc,
        "synthetic_sc_greedy_acc": acc_greedy,
        "synthetic_sc_sample_acc": acc_sample,
        "synthetic_sc_gain": acc_sc - acc_greedy,
        "torch_available": 1.0,
    }
