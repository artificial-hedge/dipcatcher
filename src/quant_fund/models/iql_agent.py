"""IQL: implicit Q-learning via expectile regression.

Kostrikov et al. 2022: fit the value function to an expectile of the
targets (tau ~ 0.7-0.9 recovers max-like behavior without querying
OOD actions), then extract the policy by advantage-weighted
behavior cloning. Fully offline.

Bench: mixed-quality position MDP data. Metric: SYNTHETIC reward of
the extracted policy vs dataset mean and vs unweighted BC.
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
        raise ImportError("iql_agent requires the `nn` extra (make sync)") from exc


def _env_step(state: FloatArray, a: int, rng: np.random.Generator) -> tuple[FloatArray, float]:
    r_prev, r_prev2, pos, sig = state
    new_pos = a - 1.0
    r_next = 0.6 * np.tanh(2.0 * sig) + 0.3 * rng.standard_normal()
    reward = new_pos * r_next - 0.05 * abs(new_pos - pos)
    new_sig = 0.7 * sig + 0.4 * rng.standard_normal()
    return np.array([r_next, r_prev, new_pos, new_sig]), float(reward)


def synth_offline(
    n_ep: int, horizon: int, rng: np.random.Generator, expert_frac: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rows_s, rows_a, rows_r, rows_n = [], [], [], []
    for e in range(n_ep):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        is_exp = e < int(n_ep * expert_frac)
        for _t in range(horizon):
            if is_exp:
                a = int(np.sign(s[3])) + 1 if rng.random() < 0.75 else int(rng.integers(3))
            else:
                a = int(rng.integers(3))
            sn, r = _env_step(s, a, rng)
            rows_s.append(s.copy())
            rows_a.append(a)
            rows_r.append(r)
            rows_n.append(sn.copy())
            s = sn
    return np.array(rows_s), np.array(rows_a), np.array(rows_r), np.array(rows_n)


def _eval_pi(pi: Any, rng: np.random.Generator, episodes: int = 24, horizon: int = 24) -> float:
    tot = 0.0
    for _e in range(episodes):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        for _t in range(horizon):
            torch = _torch()
            with torch.no_grad():
                logits = pi(torch.tensor(s, dtype=torch.float32)[None])
            a = int(logits.argmax(-1).item())
            s, r = _env_step(s, a, rng)
            tot += r
    return tot / episodes


def bench_iql_agent(
    seed: int = 20261231,
    n_ep: int = 160,
    horizon: int = 24,
    iters: int = 900,
    tau: float = 0.8,
    beta_awr: float = 3.0,
) -> dict[str, float]:
    """Expectile IQL extraction vs unweighted behavior cloning."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    s, a, r, sn = synth_offline(n_ep, horizon, rng, expert_frac=0.3)
    S = torch.tensor(s, dtype=torch.float32)
    A = torch.tensor(a, dtype=torch.long)
    Rw = torch.tensor(r, dtype=torch.float32)
    SN = torch.tensor(sn, dtype=torch.float32)

    q = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    v = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    pi = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    params = torch.nn.ModuleList([q, v, pi])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    for _i in range(iters):
        with torch.no_grad():
            q_sa = q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        diff = q_sa - v(S).squeeze(1)
        w = torch.where(diff > 0, tau, 1 - tau)
        v_loss = torch.mean(w * diff**2)
        with torch.no_grad():
            tgt = Rw + 0.95 * v(SN).squeeze(1)
        td = torch.mean((q(S).gather(1, A.unsqueeze(1)).squeeze(1) - tgt) ** 2)
        with torch.no_grad():
            adv = q_sa - v(S).squeeze(1)
            aw = torch.clamp(torch.exp(beta_awr * adv), max=50.0)
        ce = torch.nn.functional.cross_entropy(pi(S), A, reduction="none")
        awr = torch.mean(aw * ce)
        loss = v_loss + td + awr
        opt.zero_grad()
        loss.backward()
        opt.step()

    rew_iql = _eval_pi(pi, rng)
    bc = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    pb = torch.nn.ModuleList([bc])
    optb = torch.optim.Adam(pb.parameters(), lr=1e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(bc(S), A)
        optb.zero_grad()
        loss.backward()
        optb.step()
    rew_bc = _eval_pi(bc, rng)
    rew_dataset = float(r.sum() / n_ep)
    return {
        "synthetic_iql_reward": rew_iql,
        "synthetic_iql_bc_reward": rew_bc,
        "synthetic_iql_dataset_reward": rew_dataset,
        "synthetic_iql_margin_vs_bc": rew_iql - rew_bc,
        "synthetic_iql_margin_vs_dataset": rew_iql - rew_dataset,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_iql_agent()))
