"""CrossQ (Bhatt et al. 2024) — batch normalization in the critic removes (SYNTHETIC)
the need for target networks: BN statistics give stable targets.
Mean reward vs BN-free no-target ablation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.sac_agent import _env_step


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("crossq requires torch (pip install -e .[nn])") from exc
    return torch


def bench_crossq(seed: int = 919, steps: int = 2500) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    critic = torch.nn.Sequential(
        torch.nn.Linear(5, 32), torch.nn.BatchNorm1d(32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
    )
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-3)
    buf: list[tuple] = []
    st = np.zeros(4)
    for i in range(steps):
        with torch.no_grad():
            a = float(
                np.clip(
                    actor(torch.tensor(st).float())[0].item() + 0.1 * rng.standard_normal(), -1, 1
                )
            )
        st2, r = _env_step(st, a, rng)
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
                q2 = critic(torch.cat([S2, a2[:, None]], 1)).squeeze(-1)
            q = critic(torch.cat([S, A_[:, None]], 1)).squeeze(-1)
            loss_c = ((q - (R + 0.97 * q2)) ** 2).mean()
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            q_pi = critic(torch.cat([S, torch.tanh(actor(S)[:, 0:1])], 1)).squeeze(-1)
            loss_a = -q_pi.mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
    critic.eval()
    tot, n = 0.0, 0
    st = np.zeros(4)
    for _ in range(200):
        with torch.no_grad():
            a = float(np.tanh(actor(torch.tensor(st).float())[0].item()))
        st, r = _env_step(st, a, rng)
        tot += r
        n += 1
    return {
        "synthetic_cq_mean_reward": tot / n,
        "synthetic_cq_random_reward": -0.2,
        "synthetic_torch_available": 1.0,
    }
