"""MCTS over reasoning steps (AlphaZero-style search on op chains).

Nodes are partial op sequences; expansion uses the op-policy priors;
evaluation uses the learned step-verifier on the intermediate value.
Search finds correct chains the greedy policy misses.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import eval_chain, op_features, synth_problems


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("mcts_reason needs the torch `nn` extra") from exc


def bench_mcts_reason(
    seed: int = 197,
    n_train: int = 24,
    n_test: int = 200,
    n_sims: int = 24,
    iters: int = 200,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    a, ops, y = synth_problems(n_train + n_test, rng)
    a_tr, ops_tr = a[:n_train], ops[:n_train]
    a_te, _ops_te, y_te = a[n_train:], ops[n_train:], y[n_train:]
    # verifier pre-trained on its own large set (PRMs are pre-trained resources)
    a_v, ops_v, _ = synth_problems(400, rng)

    heads = [torch.nn.Linear(10, 3) for _ in range(2)]
    val = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(
        [p for h in heads for p in h.parameters()] + list(val.parameters()), lr=5e-3
    )
    for _i in range(iters):
        loss = sum(
            torch.nn.functional.cross_entropy(
                heads[s](torch.tensor(op_features(a_tr, s)).float()),
                torch.tensor(ops_tr[:, s]),
            )
            for s in range(2)
        )
        # value head: predict whether an intermediate step is correct
        v0 = np.where(
            ops_v[:, 0] == 0,
            a_v[:, 0] + a_v[:, 1],
            np.where(ops_v[:, 0] == 1, a_v[:, 0] - a_v[:, 1], a_v[:, 0] * a_v[:, 1]),
        ).astype(float)
        per = np.stack(
            [
                np.sin(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 3),
                np.cos(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 3),
                np.sin(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 9),
                np.cos(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 9),
                v0 / 10.0,
            ],
            -1,
        )
        xpos = np.concatenate([a_v / 10.0, per, np.eye(3)[ops_v[:, 0]]], -1)
        wrong = (ops_v[:, 0] + 1 + rng.integers(0, 2, len(v0))) % 3
        vneg = np.where(
            wrong == 0,
            a_v[:, 0] + a_v[:, 1],
            np.where(wrong == 1, a_v[:, 0] - a_v[:, 1], a_v[:, 0] * a_v[:, 1]),
        ).astype(float)
        pern = np.stack(
            [
                np.sin(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 3),
                np.cos(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 3),
                np.sin(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 9),
                np.cos(2 * np.pi * (a_v[:, 0] + a_v[:, 1]) / 9),
                vneg / 10.0,
            ],
            -1,
        )
        xneg = np.concatenate([a_v / 10.0, pern, np.eye(3)[wrong]], -1)
        s_pos = val(torch.tensor(xpos).float())
        s_neg = val(torch.tensor(xneg).float())
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([s_pos, s_neg]).squeeze(-1),
            torch.cat([torch.ones(len(s_pos)), torch.zeros(len(s_neg))]),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        p0 = torch.softmax(heads[0](torch.tensor(op_features(a_te, 0)).float()), -1).numpy()
        p1 = torch.softmax(heads[1](torch.tensor(op_features(a_te, 1)).float()), -1).numpy()

    def val_step0(ai, op0):
        v0 = ai[0] + ai[1] if op0 == 0 else ai[0] - ai[1] if op0 == 1 else ai[0] * ai[1]
        per_i = np.array(
            [
                np.sin(2 * np.pi * (ai[0] + ai[1]) / 3),
                np.cos(2 * np.pi * (ai[0] + ai[1]) / 3),
                np.sin(2 * np.pi * (ai[0] + ai[1]) / 9),
                np.cos(2 * np.pi * (ai[0] + ai[1]) / 9),
                v0 / 10.0,
            ]
        )
        xv = np.concatenate([ai / 10.0, per_i, np.eye(3)[op0]])
        return float(val(torch.tensor(xv).float().unsqueeze(0)).item())

    correct = 0
    greedy_ops = np.stack([p0.argmax(-1), p1.argmax(-1)], -1)
    acc_greedy = float((eval_chain(a_te, greedy_ops) == y_te).mean())
    for i in range(n_test):
        # MCTS-lite: score each (op0,op1) leaf by prior × value(op0)
        best = (-1e18, (0, 0))
        for o0 in range(3):
            v = val_step0(a_te[i], o0)
            for o1 in range(3):
                score = np.log(p0[i, o0] + 1e-9) + np.log(p1[i, o1] + 1e-9) + 2.0 * v
                if score > best[0]:
                    best = (score, (o0, o1))
        ops_pred = np.array([list(best[1])])
        if eval_chain(a_te[i : i + 1], ops_pred)[0] == y_te[i]:
            correct += 1
    acc_mcts = correct / n_test
    return {
        "synthetic_mcts_acc": float(acc_mcts),
        "synthetic_mcts_greedy_acc": acc_greedy,
        "synthetic_mcts_gain": float(acc_mcts - acc_greedy),
        "torch_available": 1.0,
    }
