"""Expert iteration (Anthony et al. 2017) — tic-tac-toe: apprentice
policy net imitates a strong search expert (minimax oracle chosen as
the expert — TTT admits exact search); self-play states are relabeled
by the expert's argmax + outcome value. Non-loss vs oracle/random.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import (
    encode_ttt,
    optimal_move,
    ttt_eval,
    ttt_legal,
    ttt_winner,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("expert_iteration requires torch (pip install -e .[nn])") from exc
    return torch


def bench_expert_iteration(seed: int = 2725, rounds: int = 30) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(9, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 9),
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    rng = np.random.default_rng(seed)
    buf: list[tuple[np.ndarray, int]] = []

    for _ in range(rounds):
        # apprentice self-play (softmax action sampling), expert relabels
        s: tuple[int, ...] = (0,) * 9
        player = 1
        while ttt_winner(s) is None:
            legal = ttt_legal(s)
            buf.append((encode_ttt(s, player), optimal_move(s, player)))
            with torch.no_grad():
                p = net(torch.tensor(encode_ttt(s, player)).float()).softmax(-1).numpy()
            a = legal[int(rng.choice(len(legal), p=p[legal] / p[legal].sum()))]
            ns = list(s)
            ns[a] = player
            s = tuple(ns)
            player = -player
        X = torch.tensor(np.array([b[0] for b in buf[-600:]])).float()
        Y = torch.tensor([b[1] for b in buf[-600:]])
        for _ in range(15):
            loss = torch.nn.functional.cross_entropy(net(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def policy(e: np.ndarray, legal: list[int]) -> int:
        with torch.no_grad():
            p = net(torch.tensor(e).float()).softmax(-1).numpy()
        return legal[int(np.argmax(p[legal]))]

    nl_or, nl_rd = ttt_eval(policy, games=60)
    return {
        "synthetic_exit_nonloss_oracle": nl_or,
        "synthetic_exit_nonloss_random": nl_rd,
        "torch_available": 1.0,
    }
