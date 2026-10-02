"""PATE — Private Aggregation of Teacher Ensembles (Papernot 2018).

K teachers vote on unlabeled queries; noisy argmax (Laplace) labels
are revealed to a student. Privacy vs utility: label agreement and
student accuracy under vote noise vs clean vote.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import acc_of, make_mlp, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("pate_teacher requires torch (pip install -e .[nn])") from exc
    return torch


def bench_pate_teacher(
    seed: int = 439,
    n: int = 400,
    n_teachers: int = 5,
    iters: int = 40,
    noise: float = 0.3,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    # teachers: disjoint data partitions
    parts = np.array_split(np.random.default_rng(seed).permutation(len(x_tr)), n_teachers)
    teachers = []
    for k, idx in enumerate(parts):
        torch.manual_seed(seed + k)
        net = make_mlp(torch)
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _i in range(iters):
            loss = torch.nn.functional.cross_entropy(net(x_tr_t[idx]), y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        teachers.append(net)
    # vote on an unlabeled query pool (train features) — m capped
    m = min(100, len(x_tr))
    votes = np.zeros((m, 2))
    for t in teachers:
        with torch.no_grad():
            votes[np.arange(m), t(x_tr_t[:m]).argmax(1).numpy()] += 1
    clean = votes.argmax(1)
    noisy_votes = votes + rng.laplace(0, noise, votes.shape)
    noisy = noisy_votes.argmax(1)
    agree = float((noisy == clean).mean())
    # student trained on noisy labels
    torch.manual_seed(seed + 99)
    stu = make_mlp(torch)
    opt = torch.optim.Adam(stu.parameters(), lr=0.02)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(stu(x_te_t[:m]), torch.tensor(noisy))
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_stu = acc_of(torch, stu, x_te_t[m:], y_te_t[m:])
    acc_teacher = float(np.mean([acc_of(torch, t, x_te_t, y_te_t) for t in teachers]))
    return {
        "synthetic_pate_noisy_agree": agree,
        "synthetic_pate_student_acc": acc_stu,
        "synthetic_pate_teacher_acc": acc_teacher,
        "torch_available": 1.0,
    }
