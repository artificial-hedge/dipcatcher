"""Neural Fictitious Self-Play (Heinrich & Silver 2016) — Kuhn poker:
each player keeps a best-response net Q (DQN-style, ε-greedy) and an
average-policy net π trained supervised on the actions Q took. π is
the deployable strategy; exploitability vs random baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import (
    CARDS,
    encode_kuhn,
    kuhn_exploit,
    kuhn_terminal,
    kuhn_util,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("nfsp requires torch (pip install -e .[nn])") from exc
    return torch


def bench_nfsp(
    seed: int = 2713, iters: int = 150, eps: float = 0.15, eta: float = 0.25
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def mnet():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    Q = [mnet(), mnet()]
    Pi = [mnet(), mnet()]
    oq = [torch.optim.Adam(n.parameters(), lr=0.01) for n in Q]
    opi = [torch.optim.Adam(n.parameters(), lr=0.01) for n in Pi]
    q_buf: list[list[tuple[np.ndarray, int, float]]] = [[], []]
    pi_buf: list[list[tuple[np.ndarray, int]]] = [[], []]

    def episode(c1: int, c2: int) -> None:
        h = ""
        traj: list[tuple[int, int, str, int]] = []  # (player, card, hist, action)
        while not kuhn_terminal(h):
            p = len(h) % 2
            c = c1 if p == 0 else c2
            e = torch.tensor(encode_kuhn(p, c, h)).float()
            with torch.no_grad():
                qv = Q[p](e).numpy()
                pv = Pi[p](e).softmax(-1).numpy()
            # anticipatory mixture: w.p. η follow avg policy, else ε-greedy BR
            if rng.random() < eta:
                a = int(rng.choice(2, p=pv))
            else:
                a = int(np.argmax(qv)) if rng.random() > eps else int(rng.integers(2))
            traj.append((p, c, h, a))
            pi_buf[p].append((encode_kuhn(p, c, h), int(np.argmax(qv))))
            h += ["p", "b"][a]
        u = kuhn_util(h, c1, c2)
        for p, c, hh, a in traj:
            q_buf[p].append((encode_kuhn(p, c, hh), a, u if p == 0 else -u))

    for _ in range(iters):
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 != c2:
                    episode(c1, c2)
        for p in (0, 1):
            if q_buf[p]:
                b = q_buf[p][-4000:]
                X = torch.tensor(np.array([t[0] for t in b])).float()
                A = torch.tensor([t[1] for t in b])
                Rr = torch.tensor([t[2] for t in b]).float()
                for _ in range(20):
                    pred = Q[p](X).gather(1, A[:, None]).squeeze(1)
                    loss = ((pred - Rr) ** 2).mean()
                    oq[p].zero_grad()
                    loss.backward()
                    oq[p].step()
            if pi_buf[p]:
                bp = pi_buf[p][-4000:]
                X = torch.tensor(np.array([t[0] for t in bp])).float()
                Y = torch.tensor([t[1] for t in bp])
                for _ in range(20):
                    logits = Pi[p](X)
                    loss = torch.nn.functional.cross_entropy(logits, Y)
                    opi[p].zero_grad()
                    loss.backward()
                    opi[p].step()

    def strat(p: int, c: int, h: str) -> np.ndarray:
        e = torch.tensor(encode_kuhn(p, c, h)).float()
        with torch.no_grad():
            return np.asarray(Pi[p](e).softmax(-1).numpy())

    expl = kuhn_exploit(strat)
    ref = kuhn_exploit(lambda p, c, h: np.array([0.5, 0.5]))
    return {
        "synthetic_nfsp_expl": expl,
        "synthetic_random_expl": ref,
        "synthetic_nfsp_drop": ref - expl,
        "synthetic_torch_available": 1.0,
    }
