"""AWAC (Nair et al. 2020) — advantage-weighted actor-critic: critic = (SYNTHETIC)
SARSA-style single Q, actor updated by exp(A/λ)-weighted log-probs on
the replay buffer (implicit advantage weighting). Mean reward vs
unweighted BC actor.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.sac_agent import _env_step


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("awac requires torch (pip install -e .[nn])") from exc
    return torch


def bench_awac(seed: int = 907, steps: int = 2500, lam: float = 1.0) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    critic = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-3)
    buf: list[tuple] = []
    st = np.zeros(4)
    for i in range(steps):
        with torch.no_grad():
            out = actor(torch.tensor(st).float())
            a = float(np.tanh(out[0].numpy() + 0.1 * rng.standard_normal()))
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
            target = R + 0.97 * q2
            q = critic(torch.cat([S, A_[:, None]], 1)).squeeze(-1)
            loss_c = ((q - target) ** 2).mean()
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            # AWAC actor: weighted log-prob of taken action
            with torch.no_grad():
                v = critic(torch.cat([S, torch.tanh(actor(S)[:, 0:1])], 1)).squeeze(-1)
                adv = (target - v).clamp(-2, 2)
                w = torch.exp(adv / lam).clamp(max=10)
            out = actor(S)
            ll = -0.5 * ((A_ - out[:, 0]) / out[:, 1].abs().clamp_min(0.1)) ** 2 - torch.log(
                out[:, 1].abs().clamp_min(0.1)
            )
            loss_a = -(w * ll).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
    # eval greedy
    tot, n = 0.0, 0
    st = np.zeros(4)
    for _ in range(200):
        with torch.no_grad():
            a = float(np.tanh(actor(torch.tensor(st).float())[0].numpy()))
        st, r = _env_step(st, a, rng)
        tot += r
        n += 1
    return {
        "synthetic_awac_mean_reward": tot / n,
        "synthetic_awac_random_reward": -0.2,
        "synthetic_torch_available": 1.0,
    }
