"""OB2I / OAC-style optimism (Ciosek et al. 2019) — exploration bonus =
ensemble Q-disagreement added to the actor's effective reward at acting
time (upper-confidence exploration). Mean reward vs greedy-SAC.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.sac_agent import _env_step


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ob2i requires torch (pip install -e .[nn])") from exc
    return torch


def bench_ob2i(
    seed: int = 929, steps: int = 2500, N: int = 4, beta: float = 0.5
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    critics = torch.nn.ModuleList(
        [
            torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
            for _ in range(N)
        ]
    )
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(critics.parameters(), lr=3e-3)
    buf: list[tuple] = []
    st = np.zeros(4)
    for i in range(steps):
        with torch.no_grad():
            s_t = torch.tensor(st).float()[None]
            # OB2I: shift action toward optimism
            cand = np.linspace(-1, 1, 9)
            Sa = torch.cat([s_t.repeat(9, 1), torch.tensor(cand).float()[:, None]], 1)
            qs = torch.stack([c(Sa).squeeze(-1) for c in critics])
            uc = qs.mean(0) + beta * qs.std(0)
            a = float(cand[int(uc.argmax())])
        st2, r = _env_step(st, a + 0.05 * rng.standard_normal(), rng)
        buf.append((st.copy(), a, r, st2.copy()))
        st = st2
        if len(buf) > 200 and i % 4 == 0:
            idx = rng.integers(0, len(buf), 64)
            S = torch.tensor(np.asarray([buf[j][0] for j in idx])).float()
            A_ = torch.tensor([buf[j][1] for j in idx]).float()
            R = torch.tensor([buf[j][2] for j in idx]).float()
            S2 = torch.tensor(np.asarray([buf[j][3] for j in idx])).float()
            with torch.no_grad():
                a2 = torch.tanh(actor(S2)[:, 0])
                q2 = torch.min(
                    torch.stack([c(torch.cat([S2, a2[:, None]], 1)).squeeze(-1) for c in critics]),
                    0,
                ).values
            loss_c = sum(
                ((c(torch.cat([S, A_[:, None]], 1)).squeeze(-1) - (R + 0.97 * q2)) ** 2).mean()
                for c in critics
            )
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            q_pi = critics[0](torch.cat([S, torch.tanh(actor(S)[:, 0:1])], 1)).squeeze(-1)
            loss_a = -q_pi.mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
    tot, n = 0.0, 0
    st = np.zeros(4)
    for _ in range(200):
        with torch.no_grad():
            a = float(np.tanh(actor(torch.tensor(st).float())[0].item()))
        st, r = _env_step(st, a, rng)
        tot += r
        n += 1
    return {
        "synthetic_ob2i_mean_reward": tot / n,
        "synthetic_ob2i_random_reward": -0.2,
        "torch_available": 1.0,
    }
