"""VDN (Sunehag et al., 2017) — value decomposition networks: (SYNTHETIC)
joint Q = Σ_i Q_i(obs_i, a_i) learned end-to-end by TD on the joint
return. Includes the shared 2-agent ring-rendezvous environment used
by the wave-119 MARL modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class RendezvousEnv:
    """2 agents on a ring of `n_pos` cells; obs_i = (x_i, x_j).
    Actions: 0 stay, 1 CW, 2 CCW. Reward +10 on meeting (episode
    ends), −1 otherwise; horizon `horizon`."""

    def __init__(self, n_pos: int = 5, horizon: int = 15):
        self.n_pos = n_pos
        self.horizon = horizon
        self.n_agents = 2
        self.n_actions = 3
        self.x = np.zeros(2, dtype=np.int64)
        self.t = 0

    def reset(self, rng: np.random.Generator) -> list[FloatArray]:
        self.x = rng.integers(0, self.n_pos, 2)
        while self.x[0] == self.x[1]:
            self.x = rng.integers(0, self.n_pos, 2)
        self.t = 0
        return self.obs()

    def obs(self) -> list[FloatArray]:
        out: list[FloatArray] = [
            np.array([float(self.x[0]), float(self.x[1])]),
            np.array([float(self.x[1]), float(self.x[0])]),
        ]
        return out

    def step(self, actions: list[int]) -> tuple[list[FloatArray], float, bool]:
        moves = np.array([0, 1, -1])
        self.x = (self.x + moves[np.asarray(actions)]) % self.n_pos
        self.t += 1
        done = bool(self.x[0] == self.x[1]) or self.t >= self.horizon
        r = 10.0 if self.x[0] == self.x[1] else -1.0
        return self.obs(), r, done


def obs_index(x: FloatArray, n_pos: int) -> int:
    """Joint discrete index of a 2-agent obs pair."""
    return int(x[0]) * n_pos + int(x[1])


class VDN:
    """Tabular VDN: per-agent Q tables; joint action greedy on the
    sum. Parameters are the Q tables themselves — the decomposition
    structure is exactly additive."""

    def __init__(self, env: RendezvousEnv, lr: float = 0.1, gamma: float = 0.95, eps: float = 0.1):
        self.env = env
        self.q = [np.zeros((env.n_pos * env.n_pos, env.n_actions)) for _ in range(env.n_agents)]
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def joint_greedy(self, obs: list[FloatArray]) -> list[int]:
        s_idx = [obs_index(o, self.env.n_pos) for o in obs]
        qs = np.stack([self.q[i][s_idx[i]] for i in range(2)])
        q_sum = qs[0][:, None] + qs[1][None, :]
        a0, a1 = np.unravel_index(np.argmax(q_sum), q_sum.shape)
        return [int(a0), int(a1)]

    def train(self, episodes: int, rng: np.random.Generator) -> float:
        rets = []
        for _ in range(episodes):
            obs = self.env.reset(rng)
            ret, done = 0.0, False
            while not done:
                if rng.random() < self.eps:
                    acts = [int(rng.integers(3)), int(rng.integers(3))]
                else:
                    acts = self.joint_greedy(obs)
                nobs, r, done = self.env.step(acts)
                ret += r
                s_idx = [obs_index(o, self.env.n_pos) for o in obs]
                ns_idx = [obs_index(o, self.env.n_pos) for o in nobs]
                if done:
                    tgt = r
                else:
                    nqs = np.stack([self.q[i][ns_idx[i]] for i in range(2)])
                    tgt = r + self.gamma * float((nqs[0][:, None] + nqs[1][None, :]).max())
                # VDN: distribute the SAME TD error to each agent
                cur = sum(self.q[i][s_idx[i]][acts[i]] for i in range(2))
                td = tgt - cur
                for i in range(2):
                    self.q[i][s_idx[i]][acts[i]] += self.lr * td
                obs = nobs
            rets.append(ret)
        return float(np.mean(rets[-20:]))


def bench_vdn(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: VDN learns the rendezvous coordination — meeting
    return approaches the +8-9 optimal from the −15 baseline."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = RendezvousEnv()
    agent = VDN(env, eps=0.15)
    ret = agent.train(600, rng)
    out["synthetic_vdn_return"] = ret
    out["synthetic_vdn_learns"] = float(ret > 0)
    # additive decomposition check
    obs = env.reset(rng)
    s = [obs_index(o, env.n_pos) for o in obs]
    a = agent.joint_greedy(obs)
    qsum = agent.q[0][s[0]][a[0]] + agent.q[1][s[1]][a[1]]
    qmat = agent.q[0][s[0]][:, None] + agent.q[1][s[1]][None, :]
    out["synthetic_vdn_greedy_max"] = float(abs(qsum - qmat.max()) < 1e-9)
    return out


if __name__ == "__main__":
    print(bench_vdn())
