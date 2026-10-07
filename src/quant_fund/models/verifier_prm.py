"""Process-reward verifier (Lightman et al. 2024) (SYNTHETIC).

A verifier trained on labelled intermediate states re-scores K
sampled solution candidates by step-correctness; argmax-PRM beats
single-sample and majority vote on the op-chain fixture.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import eval_chain, op_features, synth_problems


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("verifier_prm needs the torch `nn` extra") from exc


def bench_verifier_prm(
    seed: int = 193,
    n_train: int = 24,
    n_test: int = 300,
    k: int = 12,
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
    prm = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(
        [p for h in heads for p in h.parameters()] + list(prm.parameters()), lr=5e-3
    )
    for _i in range(iters):
        loss = sum(
            torch.nn.functional.cross_entropy(
                heads[s](torch.tensor(op_features(a_tr, s)).float()),
                torch.tensor(ops_tr[:, s]),
            )
            for s in range(2)
        )
        # verifier on intermediate state after step-0 op application
        v0 = np.where(
            ops_v[:, 0] == 0,
            a_v[:, 0] + a_v[:, 1],
            np.where(ops_v[:, 0] == 1, a_v[:, 0] - a_v[:, 1], a_v[:, 0] * a_v[:, 1]),
        ).astype(float)
        oh_p = np.eye(3)[ops_v[:, 0]]
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
        xpos = np.concatenate([a_v / 10.0, per, oh_p], -1)
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
        s_pos = prm(torch.tensor(xpos).float())
        s_neg = prm(torch.tensor(xneg).float())
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([s_pos, s_neg]).squeeze(-1),
            torch.cat([torch.ones(len(s_pos)), torch.zeros(len(s_neg))]),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sampled = np.zeros((n_test, k, 2), dtype=np.int64)
        for s in range(2):
            probs = torch.softmax(heads[s](torch.tensor(op_features(a_te, s)).float()), -1)
            sampled[:, :, s] = np.array(
                [np.array(torch.multinomial(probs[i], k, replacement=True)) for i in range(n_test)]
            )
        scores = np.zeros((n_test, k))
        for i in range(n_test):
            for j in range(k):
                v0 = np.where(
                    sampled[i, j, 0] == 0,
                    a_te[i, 0] + a_te[i, 1],
                    np.where(
                        sampled[i, j, 0] == 1, a_te[i, 0] - a_te[i, 1], a_te[i, 0] * a_te[i, 1]
                    ),
                )
                per_i = np.array(
                    [
                        np.sin(2 * np.pi * (a_te[i, 0] + a_te[i, 1]) / 3),
                        np.cos(2 * np.pi * (a_te[i, 0] + a_te[i, 1]) / 3),
                        np.sin(2 * np.pi * (a_te[i, 0] + a_te[i, 1]) / 9),
                        np.cos(2 * np.pi * (a_te[i, 0] + a_te[i, 1]) / 9),
                        v0 / 10.0,
                    ]
                )
                xv = np.concatenate([a_te[i] / 10.0, per_i, np.eye(3)[sampled[i, j, 0]]])
                scores[i, j] = prm(torch.tensor(xv).float().unsqueeze(0)).item()
        best = sampled[np.arange(n_test), scores.argmax(-1)]
        acc_prm = float((eval_chain(a_te, best) == y_te).mean())
        acc_first = float((eval_chain(a_te, sampled[:, 0, :]) == y_te).mean())
    return {
        "synthetic_prm_acc": acc_prm,
        "synthetic_prm_single_acc": acc_first,
        "synthetic_prm_gain": acc_prm - acc_first,
        "synthetic_torch_available": 1.0,
    }
