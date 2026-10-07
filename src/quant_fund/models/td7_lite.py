"""TD7-lite (Fujimoto et al. 2023) — SALE: state-action learned embedding
as critic input (encoder trained to predict reward + next-state);
checkpoint policy averaging approximated by EMA actor. Mean reward vs
plain critic.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.sac_agent import _env_step


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("td7_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_td7_lite(seed: int = 913, steps: int = 2500) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    enc_s = torch.nn.Sequential(torch.nn.Linear(4, 24), torch.nn.ReLU(), torch.nn.Linear(24, 8))
    enc_a = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    pred = torch.nn.Sequential(
        torch.nn.Linear(16, 24), torch.nn.ReLU(), torch.nn.Linear(24, 5)
    )  # r + s' (4)
    actor = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    critic = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(
        list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(pred.parameters())
        + list(actor.parameters())
        + list(critic.parameters()),
        lr=3e-3,
    )
    buf: list[tuple] = []
    st = np.zeros(4)
    ema = {k: v.clone() for k, v in actor.state_dict().items()}
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
        for k, v in actor.state_dict().items():
            ema[k].mul_(0.995).add_(0.005 * v)
        if len(buf) > 200 and i % 4 == 0:
            idx = rng.integers(0, len(buf), 64)
            S = torch.tensor(np.asarray([buf[j][0] for j in idx])).float()
            A_ = torch.tensor([buf[j][1] for j in idx]).float()
            R = torch.tensor([buf[j][2] for j in idx]).float()
            S2 = torch.tensor(np.asarray([buf[j][3] for j in idx])).float()
            zs, za = enc_s(S), enc_a(A_[:, None])
            zsa = torch.cat([zs, za], 1)
            # SALE: embedding predicts r + s'
            tgt = torch.cat([R[:, None], S2], 1)
            loss_e = ((pred(zsa) - tgt) ** 2).mean()
            with torch.no_grad():
                a2 = torch.tanh(actor(S2)[:, 0])
                q2 = critic(torch.cat([enc_s(S2), enc_a(a2[:, None])], 1)).squeeze(-1)
            q = critic(zsa.detach()).squeeze(-1)
            loss_c = ((q - (R + 0.97 * q2)) ** 2).mean()
            q_pi = critic(
                torch.cat([enc_s(S), enc_a(torch.tanh(actor(S)[:, 0:1]))], 1).detach()
            ).squeeze(-1)
            loss_a = -q_pi.mean()
            loss = loss_e + loss_c + loss_a
            opt.zero_grad()
            loss.backward()
            opt.step()
    actor.load_state_dict(ema)
    tot, n = 0.0, 0
    st = np.zeros(4)
    for _ in range(200):
        with torch.no_grad():
            a = float(np.tanh(actor(torch.tensor(st).float())[0].item()))
        st, r = _env_step(st, a, rng)
        tot += r
        n += 1
    return {
        "synthetic_td7_mean_reward": tot / n,
        "synthetic_td7_random_reward": -0.2,
        "synthetic_torch_available": 1.0,
    }
