"""CQL: conservative Q-learning for offline RL.

Kumar et al. 2020: penalize the Q-values of out-of-distribution
actions (logsumexp over actions minus dataset actions) so the learned
policy doesn't extrapolate to actions the data never tried.

Bench: same position MDP as decision_transformer (mixed-quality
offline data). Metric: SYNTHETIC mean reward of the argmax-Q policy
vs the dataset-average policy and vs plain (non-conservative)
Q-learning — CQL should be more robust when data is mostly bad.
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
        raise ImportError("cql_agent requires the `nn` extra (make sync)") from exc


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
    """(N,4) states, (N,) actions, (N,) rewards, (N,4) next states."""
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
    return (
        np.array(rows_s),
        np.array(rows_a),
        np.array(rows_r),
        np.array(rows_n),
    )


def _eval_policy(
    qnet: Any, rng: np.random.Generator, episodes: int = 24, horizon: int = 24
) -> float:
    import torch  # noqa: F401

    tot = 0.0
    for _e in range(episodes):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        for _t in range(horizon):
            with torch.no_grad():
                q = qnet(torch.tensor(s, dtype=torch.float32)[None])
            a = int(q.argmax(-1).item())
            s, r = _env_step(s, a, rng)
            tot += r
    return tot / episodes


def bench_cql_agent(
    seed: int = 20261231,
    n_ep: int = 160,
    horizon: int = 24,
    iters: int = 900,
    alpha_cql: float = 1.0,
) -> dict[str, float]:
    """CQL vs plain fitted-Q on mostly-random offline data."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    s, a, r, sn = synth_offline(n_ep, horizon, rng, expert_frac=0.3)
    S = torch.tensor(s, dtype=torch.float32)
    A = torch.tensor(a, dtype=torch.long)
    Rw = torch.tensor(r, dtype=torch.float32)
    SN = torch.tensor(sn, dtype=torch.float32)

    def fit(conservative: float) -> Any:
        q = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
        params = torch.nn.ModuleList([q])
        opt = torch.optim.Adam(params.parameters(), lr=1e-3)
        for _i in range(iters):
            with torch.no_grad():
                tgt = Rw + 0.95 * q(SN).max(-1).values
            qs = q(S)
            td = torch.mean((qs.gather(1, A.unsqueeze(1)).squeeze(1) - tgt) ** 2)
            penalty = torch.logsumexp(qs, 1).mean() - qs.gather(1, A.unsqueeze(1)).mean()
            loss = td + conservative * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
        return q

    q_cql = fit(alpha_cql)
    q_plain = fit(0.0)
    rew_cql = _eval_policy(q_cql, rng)
    rew_plain = _eval_policy(q_plain, rng)
    rew_dataset = float(r.sum() / n_ep)
    return {
        "synthetic_cql_reward": rew_cql,
        "synthetic_cql_plainq_reward": rew_plain,
        "synthetic_cql_dataset_reward": rew_dataset,
        "synthetic_cql_margin_vs_plain": rew_cql - rew_plain,
        "synthetic_cql_margin_vs_dataset": rew_cql - rew_dataset,
        "torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_cql_agent()))
