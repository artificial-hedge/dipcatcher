"""Decision Transformer: return-conditioned sequence policy.

Chen et al. 2021: cast RL as sequence modeling — a causal transformer
over (return-to-go, state, action) tokens; at test time condition on a
high target return and the model imitates the trajectories that
achieved it. Offline; no bootstrapping.

Bench: synthetic position MDP (action = target position in {-1,0,1},
reward = pos * next_ret - cost * |dpos|) with a learnable momentum
signal. Mixed-quality offline data (noisy expert + random). Metric:
out-of-sample mean reward vs dataset-mean policy and random —
SYNTHETIC, labeled.
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
        raise ImportError("decision_transformer requires the `nn` extra (make sync)") from exc


def _env_step(state: FloatArray, a: int, rng: np.random.Generator) -> tuple[FloatArray, float, int]:
    """state = [r_prev, r_prev2, pos, signal]; a in {0,1,2} -> pos-1."""
    r_prev, r_prev2, pos, sig = state
    new_pos = a - 1.0
    r_next = 0.6 * np.tanh(2.0 * sig) + 0.3 * rng.standard_normal()
    reward = new_pos * r_next - 0.05 * abs(new_pos - pos)
    new_sig = 0.7 * sig + 0.4 * rng.standard_normal()
    return np.array([r_next, r_prev, new_pos, new_sig]), float(reward), 0


def synth_trajectories(
    n_ep: int, horizon: int, rng: np.random.Generator, expert_frac: float = 0.7
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Mixed-quality offline data: expert follows sign(signal) w.p. 0.8."""
    states = np.zeros((n_ep, horizon, 4))
    acts = np.zeros((n_ep, horizon))
    rews = np.zeros((n_ep, horizon))
    for e in range(n_ep):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        is_exp = e < int(n_ep * expert_frac)
        for t in range(horizon):
            if is_exp:
                a = int(np.sign(s[3])) + 1 if rng.random() < 0.8 else int(rng.integers(3))
            else:
                a = int(rng.integers(3))
            s, r, _ = _env_step(s, a, rng)
            states[e, t], acts[e, t], rews[e, t] = s, a, r
    return states, acts, rews


def bench_decision_transformer(
    seed: int = 20261231,
    n_ep: int = 200,
    horizon: int = 24,
    iters: int = 1600,
) -> dict[str, float]:
    """Return-conditioned transformer vs dataset behavior vs random."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    states, acts, rews = synth_trajectories(n_ep, horizon, rng)
    rtg = np.flip(np.cumsum(np.flip(rews, 1).copy(), 1), 1).copy()  # return-to-go
    S = torch.tensor(states, dtype=torch.float32)
    A = torch.tensor(acts, dtype=torch.long)
    R = torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1)

    d = 48
    emb_s = torch.nn.Linear(4, d)
    emb_a = torch.nn.Embedding(3, d)
    emb_r = torch.nn.Linear(1, d)
    pos_emb = torch.nn.Parameter(torch.randn(1, horizon * 3, d) * 0.02)
    trf = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d, 4, 64, batch_first=True), 2
    )
    head = torch.nn.Linear(d, 3)
    params = torch.nn.ModuleList([emb_s, emb_a, emb_r, trf, head])
    opt = torch.optim.Adam(list(params.parameters()) + [pos_emb], lr=2e-3)
    mask = torch.triu(torch.ones(horizon * 3, horizon * 3), 1).bool()

    def seq_tokens(rt: Any, s: Any, a: Any) -> Any:
        B = s.shape[0]
        toks = torch.stack([emb_r(rt), emb_s(s), emb_a(a)], 2).reshape(B, horizon * 3, d)
        return toks + pos_emb

    for _ in range(iters):
        toks = seq_tokens(R, S, A)
        h = trf(toks, mask=mask)
        # action predicted at the state-token position (idx 3t+1)
        logits = head(h[:, 1::3])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), A.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def dt_reward(target: float, episodes: int = 24) -> float:
        tot = 0.0
        for _e in range(episodes):
            s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
            rt_seq, s_seq, a_seq = [], [], []
            cum = 0.0
            for _t in range(horizon):
                rt_seq.append(target - cum)
                s_seq.append(s.copy())
                rt_t = torch.tensor(rt_seq, dtype=torch.float32)[None, :, None]
                s_t = torch.tensor(np.array(s_seq), dtype=torch.float32)[None]
                a_pad = torch.zeros(1, len(s_seq), dtype=torch.long)
                toks = (
                    torch.stack([emb_r(rt_t), emb_s(s_t), emb_a(a_pad)], 2).reshape(1, -1, d)
                    + pos_emb[:, : 3 * len(s_seq)]
                )
                h = trf(toks)
                logits = head(h[:, -2])
                a = int(logits.argmax(-1).item())
                a_seq.append(a)
                s, r, _ = _env_step(s, a, rng)
                cum += r
            tot += cum
        return tot / episodes

    dt = dt_reward(float(rtg.max(axis=1).mean()))
    dataset_mean = float(rews.sum(axis=1).mean())
    rand = 0.0
    for _e in range(24):
        s = np.array([0.0, 0.0, 0.0, rng.standard_normal()])
        for _t in range(horizon):
            s, r, _ = _env_step(s, int(rng.integers(3)), rng)
            rand += r
    rand /= 24
    return {
        "synthetic_dt_reward": dt,
        "synthetic_dt_dataset_reward": dataset_mean,
        "synthetic_dt_random_reward": rand,
        "synthetic_dt_margin_vs_dataset": dt - dataset_mean,
        "synthetic_dt_margin_vs_random": dt - rand,
        "torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_decision_transformer()))
