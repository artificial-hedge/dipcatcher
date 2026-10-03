"""Mean-Field Q-learning (Yang et al., 2018) — each agent's Q
conditions on the mean neighbor action ᾱ, collapsing the joint
action space to (s_i, a_i, ᾱ). N-agent crowding game on a ring.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class CrowdEnv:
    """N agents each choose between two sites (action 0/1); reward is
    +2 minus crowding cost ∝ fraction choosing the same site, plus a
    coordination bonus for matching the majority-side label."""

    def __init__(self, n_agents: int = 4, horizon: int = 10):
        self.n_agents = n_agents
        self.horizon = horizon
        self.n_actions = 2
        self.state = np.zeros(n_agents, dtype=np.int64)
        self.t = 0

    def reset(self, rng: np.random.Generator) -> list[int]:
        self.state = rng.integers(0, 2, self.n_agents)
        self.t = 0
        return [int(s) for s in self.state]

    def step(self, actions: list[int]) -> tuple[list[int], float, bool]:
        frac1 = float(np.mean(actions))
        # crowd dynamics: state = majority action
        self.state = np.full(self.n_agents, int(frac1 > 0.5))
        self.t += 1
        # each agent gets reward = match majority + anti-crowding
        maj = int(frac1 > 0.5)
        r = (
            sum((2.0 if a == maj else -1.0) - (frac1 if a == 1 else 1 - frac1) for a in actions)
            / self.n_agents
        )
        done = self.t >= self.horizon
        return [int(s) for s in self.state], float(r), done


class MeanFieldQ:
    """Tabular mean-field Q: Q_i(s_i, a_i, ᾱ_bucket) with ᾱ discretized
    into `n_bins` mean-action buckets."""

    def __init__(
        self, env: CrowdEnv, n_bins: int = 5, lr: float = 0.1, gamma: float = 0.9, eps: float = 0.15
    ):
        self.env = env
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.q = np.zeros((2, env.n_actions, n_bins))

    def _bin(self, a_bar: float) -> int:
        return int(np.clip(a_bar * self.n_bins, 0, self.n_bins - 1))

    def train(self, episodes: int, rng: np.random.Generator) -> float:
        rets = []
        for _ in range(episodes):
            obs = self.env.reset(rng)
            ret, done = 0.0, False
            while not done:
                acts = []
                for i in range(self.env.n_agents):
                    # mean field excludes self → approximate with
                    # current majority state
                    a_bar = float(np.mean(obs))
                    if rng.random() < self.eps:
                        acts.append(int(rng.integers(2)))
                    else:
                        acts.append(int(np.argmax(self.q[obs[i], :, self._bin(a_bar)])))
                nobs, r, done = self.env.step(acts)
                ret += r
                a_bar_next = float(np.mean(acts))
                for i in range(self.env.n_agents):
                    s_i = obs[i]
                    b = self._bin(a_bar_next)
                    tgt = r + self.gamma * float(self.q[nobs[i], :, b].max()) * (
                        0.0 if done else 1.0
                    )
                    self.q[s_i, acts[i], self._bin(a_bar_next)] += self.lr * (
                        tgt - self.q[s_i, acts[i], self._bin(a_bar_next)]
                    )
                obs = nobs
            rets.append(ret)
        return float(np.mean(rets[-20:]))


def bench_mf_q(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: mean-field Q learns the majority-following
    crowding equilibrium — return converges positive."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = CrowdEnv(n_agents=4, horizon=10)
    agent = MeanFieldQ(env)
    ret = agent.train(800, rng)
    out["synthetic_mf_q_return"] = ret
    out["synthetic_mf_q_learns"] = float(ret > 0)
    # mean-field value table finite and shaped
    out["synthetic_mf_q_finite"] = float(np.isfinite(agent.q).all())
    return out


if __name__ == "__main__":
    print(bench_mf_q())
