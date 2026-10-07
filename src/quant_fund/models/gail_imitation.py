"""GAIL: generative adversarial imitation learning (SYNTHETIC).

Ho & Ermon 2016: a discriminator distinguishes expert from policy
state-action pairs; the policy is trained (via PPO-style clipped
updates) to maximize the discriminator's confusion reward — imitation
without a hand-crafted reward.

Bench: expert demos on the position MDP (expert follows the signal);
GAIL policy vs plain behavior cloning — GAIL should match expert
state-action occupancy more closely (higher discriminator reward,
better reward).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("gail_imitation requires the `nn` extra (make sync)") from exc


def _env_step(state: FloatArray, a: int, rng: np.random.Generator) -> tuple[FloatArray, float]:
    r_prev, r_prev2, pos, sig = state
    new_pos = a - 1.0
    r_next = 0.6 * np.tanh(2.0 * sig) + 0.3 * rng.standard_normal()
    reward = new_pos * r_next - 0.05 * abs(new_pos - pos)
    new_sig = 0.7 * sig + 0.4 * rng.standard_normal()
    return np.array([r_next, r_prev, new_pos, new_sig]), float(reward)


def synth_expert(
    n_ep: int, horizon: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    rows_s, rows_a = [], []
    for _e in range(n_ep):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        for _t in range(horizon):
            a = int(np.sign(s[3])) + 1 if rng.random() < 0.9 else int(rng.integers(3))
            rows_s.append(s.copy())
            rows_a.append(a)
            s, _r = _env_step(s, a, rng)
    return np.array(rows_s), np.array(rows_a)


def _eval(pi: Any, rng: np.random.Generator, episodes: int = 20, horizon: int = 20) -> float:
    tot = 0.0
    torch = _torch()
    for _e in range(episodes):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        for _t in range(horizon):
            with torch.no_grad():
                logits = pi(torch.tensor(s, dtype=torch.float32)[None])
            a = int(logits.argmax(-1).item())
            s, r = _env_step(s, a, rng)
            tot += r
    return tot / episodes


def bench_gail_imitation(
    seed: int = 20261231,
    n_ep: int = 120,
    horizon: int = 20,
    iters: int = 700,
) -> dict[str, float]:
    """GAIL vs behavior cloning on 90%-expert demos; SYNTHETIC."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    es, ea = synth_expert(n_ep, horizon, rng)
    ES = torch.tensor(es, dtype=torch.float32)
    EA = torch.tensor(ea, dtype=torch.long)
    one_ea = torch.nn.functional.one_hot(EA, 3).float()

    disc = torch.nn.Sequential(torch.nn.Linear(7, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    pi = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    opt_p = torch.optim.Adam(pi.parameters(), lr=3e-4)

    # BC warm-start: GAIL fine-tunes occupancy on top of a decent
    # imitation prior (standard practice — pure GAIL underfits)
    for _w in range(400):
        wloss = torch.nn.functional.cross_entropy(pi(ES), EA)
        opt_p.zero_grad()
        wloss.backward()
        opt_p.step()

    def policy_sa() -> tuple[Any, Any]:
        """Sample policy state-action pairs via short rollouts."""
        ss, aa = [], []
        for _e in range(6):
            s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
            for _t in range(horizon):
                logits = pi(torch.tensor(s, dtype=torch.float32)[None])
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
                ss.append(s.copy())
                aa.append(a)
                s, _r = _env_step(s, a, rng)
        S = torch.tensor(np.array(ss), dtype=torch.float32)
        A1 = torch.nn.functional.one_hot(torch.tensor(aa), 3).float()
        return S, A1

    for _i in range(iters):
        PS, PA = policy_sa()
        ex_in = torch.cat([ES, one_ea], 1)
        po_in = torch.cat([PS, PA], 1)
        dloss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc(ex_in).squeeze(1), torch.ones(ex_in.shape[0])
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            disc(po_in).squeeze(1), torch.zeros(po_in.shape[0])
        )
        opt_d.zero_grad()
        dloss.backward()
        opt_d.step()
        if _i % 2 == 0:
            # policy step: treat disc reward on sampled pairs as score
            PS, PA = policy_sa()
            logits = pi(PS)
            logp = torch.nn.functional.log_softmax(logits, -1)
            chosen = (logp * PA).sum(1)
            with torch.no_grad():
                rew = -torch.nn.functional.binary_cross_entropy_with_logits(
                    disc(torch.cat([PS, PA], 1)).squeeze(1),
                    torch.zeros(PS.shape[0]),
                    reduction="none",
                )
            # adversarial pull + BC anchor keeps the policy on-support
            ploss = -torch.mean(
                chosen * (rew - rew.mean())
            ) + 0.5 * torch.nn.functional.cross_entropy(logits, PA.argmax(1))
            opt_p.zero_grad()
            ploss.backward()
            opt_p.step()

    # occupancy diagnostics: expert vs policy discriminator score gap
    PS, PA = policy_sa()
    with torch.no_grad():
        ex_p = torch.sigmoid(disc(torch.cat([ES, one_ea], 1))).mean()
        po_p = torch.sigmoid(disc(torch.cat([PS, PA], 1))).mean()
    rew_gail = _eval(pi, rng)
    bc = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    pb = torch.nn.ModuleList([bc])
    optb = torch.optim.Adam(pb.parameters(), lr=1e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(bc(ES), EA)
        optb.zero_grad()
        loss.backward()
        optb.step()
    rew_bc = _eval(bc, rng)
    return {
        "synthetic_gail_reward": rew_gail,
        "synthetic_gail_bc_reward": rew_bc,
        "synthetic_gail_margin_vs_bc": rew_gail - rew_bc,
        "synthetic_gail_disc_expert_mean": float(ex_p),
        "synthetic_gail_disc_policy_mean": float(po_p),
        "synthetic_gail_disc_gap": float(ex_p - po_p),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_gail_imitation()))
