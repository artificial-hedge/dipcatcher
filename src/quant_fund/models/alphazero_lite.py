"""AlphaZero-lite (Silver et al. 2018) — tic-tac-toe self-play:
policy/value nets guide PUCT MCTS; visit distributions become the
training signal; terminal outcome the value target. Non-loss rate vs
the minimax oracle and vs random.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import (
    encode_ttt,
    ttt_eval,
    ttt_legal,
    ttt_winner,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("alphazero_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_alphazero_lite(seed: int = 2719, episodes: int = 60, sims: int = 25) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    pv = torch.nn.Sequential(
        torch.nn.Linear(9, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
    )
    pol = torch.nn.Linear(64, 9)
    val = torch.nn.Linear(64, 1)
    opt = torch.optim.Adam(
        list(pv.parameters()) + list(pol.parameters()) + list(val.parameters()), lr=0.005
    )
    rng = np.random.default_rng(seed)
    buf: list[tuple[np.ndarray, np.ndarray, float]] = []

    def pv_fn(s: tuple[int, ...], player: int) -> tuple[np.ndarray, float]:
        e = torch.tensor(encode_ttt(s, player)).float()
        with torch.no_grad():
            f = pv(e)
            return pol(f).softmax(-1).numpy(), float(torch.tanh(val(f)).item())

    def mcts_policy(s0: tuple[int, ...], player: int) -> np.ndarray:
        """PUCT search returning the root visit distribution."""
        root_legal = ttt_legal(s0)
        if len(root_legal) == 1:
            out = np.zeros(9)
            out[root_legal[0]] = 1.0
            return out
        N: dict[tuple[int, ...], np.ndarray] = {}
        W: dict[tuple[int, ...], np.ndarray] = {}
        Pr: dict[tuple[int, ...], np.ndarray] = {}
        p0, _ = pv_fn(s0, player)
        Pr[s0] = p0
        N[s0] = np.zeros(9)
        W[s0] = np.zeros(9)
        for _ in range(sims):
            s, pl = s0, player
            path: list[tuple[tuple[int, ...], int]] = []
            v = 0.0
            while True:
                w = ttt_winner(s)
                if w is not None:
                    v = 0.0 if w == 0 else -1.0  # mover at s lost (winner was -pl) or draw
                    break
                if s not in N:
                    pv_, vv = pv_fn(s, pl)
                    N[s] = np.zeros(9)
                    W[s] = np.zeros(9)
                    Pr[s] = pv_
                    v = vv
                    break
                legal = ttt_legal(s)
                q = np.divide(W[s], np.maximum(N[s], 1), out=np.zeros(9), where=N[s] > 0)
                ucb = q[legal] + 1.5 * Pr[s][legal] * np.sqrt(N[s].sum() + 1) / (1 + N[s][legal])
                a = legal[int(np.argmax(ucb))]
                path.append((s, a))
                ns = list(s)
                ns[a] = pl
                s = tuple(ns)
                pl = -pl
            for sp, ap in reversed(path):
                v = -v
                W[sp][ap] += v
                N[sp][ap] += 1
        cnt = N[s0][root_legal].astype(np.float64) + 1e-8
        pi = np.zeros(9)
        pi[root_legal] = cnt / cnt.sum()
        return pi

    for _ep in range(episodes):
        s: tuple[int, ...] = (0,) * 9
        data: list[tuple[np.ndarray, np.ndarray, int]] = []
        player = 1
        while ttt_winner(s) is None:
            pi = mcts_policy(s, player)
            data.append((encode_ttt(s, player), pi, player))
            a = int(rng.choice(9, p=pi))
            ns = list(s)
            ns[a] = player
            s = tuple(ns)
            player = -player
        w = ttt_winner(s) or 0
        for e, pi, pl in data:
            buf.append((e, pi, float(w) * pl))
        if len(buf) > 100:
            X = torch.tensor(np.array([b[0] for b in buf[-500:]])).float()
            Pt = torch.tensor(np.array([b[1] for b in buf[-500:]])).float()
            Vt = torch.tensor([b[2] for b in buf[-500:]]).float()
            f = pv(X)
            loss = (pol(f).log_softmax(-1) * Pt).sum(-1).mean() * -1 + (
                (torch.tanh(val(f)).squeeze(1) - Vt) ** 2
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def policy(e: np.ndarray, legal: list[int]) -> int:
        with torch.no_grad():
            p = pol(pv(torch.tensor(e).float())).softmax(-1).numpy()
        p = p[legal]
        return legal[int(np.argmax(p))]

    nl_or, nl_rd = ttt_eval(policy, games=40)
    return {
        "synthetic_az_nonloss_oracle": nl_or,
        "synthetic_az_nonloss_random": nl_rd,
        "torch_available": 1.0,
    }
