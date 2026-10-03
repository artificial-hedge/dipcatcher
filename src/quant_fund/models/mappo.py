"""MAPPO (Yu et al., 2021) — centralized-value PPO for cooperative
MARL: shared critic V(s_global), per-agent PPO-clipped policy updates
on advantages computed from the joint value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.coma import _softmax
from quant_fund.models.vdn import RendezvousEnv, obs_index

FloatArray = NDArray[np.float64]


class MAPPO:
    """Tabular central critic + softmax policies with PPO clipping."""

    def __init__(
        self,
        env: RendezvousEnv,
        lr_v: float = 0.1,
        lr_p: float = 0.05,
        gamma: float = 0.95,
        clip: float = 0.2,
    ):
        self.env = env
        self.gamma = gamma
        self.lr_v = lr_v
        self.lr_p = lr_p
        self.clip = clip
        n_gs = env.n_pos**env.n_agents
        n_ls = env.n_pos * env.n_pos
        self.v = np.zeros(n_gs)
        self.theta = [np.zeros((n_ls, env.n_actions)) for _ in range(2)]

    def _pi(self, i: int, o_idx: int) -> FloatArray:
        return _softmax(self.theta[i][o_idx])

    def _glob(self, obs: list[FloatArray]) -> int:
        return obs_index(obs[0], self.env.n_pos)

    def _episode(self, rng: np.random.Generator, eps: float = 0.05) -> tuple[list, float]:
        obs = self.env.reset(rng)
        traj, ret, done = [], 0.0, False
        while not done:
            o_idx = [obs_index(o, self.env.n_pos) for o in obs]
            pis = [self._pi(i, o_idx[i]) for i in range(2)]
            acts = [
                int(rng.choice(3, p=pis[i])) if rng.random() > eps else int(rng.integers(3))
                for i in range(2)
            ]
            nobs, r, done = self.env.step(acts)
            traj.append((self._glob(obs), o_idx, acts, pis, r, done))
            obs = nobs
            ret += r
        return traj, ret

    def train(self, episodes: int, rng: np.random.Generator) -> float:
        rets = []
        for _ in range(episodes):
            traj, ret = self._episode(rng)
            rets.append(ret)
            # Monte-Carlo advantage from central critic
            g = 0.0
            advs = []
            for _sg, _o, _a, _p, r, done in reversed(traj):
                g = r + self.gamma * g * (0.0 if done else 1.0)
                advs.append(g)
            advs = advs[::-1]
            for t, (sg, o_idx, acts, pis, r, done) in enumerate(traj):
                # central critic TD
                ns = traj[t + 1][0] if t + 1 < len(traj) else sg
                tgt = r + self.gamma * self.v[ns] * (0.0 if done else 1.0)
                self.v[sg] += self.lr_v * (tgt - self.v[sg])
                adv = advs[t] - self.v[sg]
                for i in range(2):
                    pi_old = pis[i]
                    pi_now = self._pi(i, o_idx[i])
                    ratio = float(pi_now[acts[i]] / (pi_old[acts[i]] + 1e-9))
                    surr = ratio * adv
                    surr_clip = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
                    if surr <= surr_clip:
                        grad = -pi_now
                        grad[acts[i]] += 1.0
                        self.theta[i][o_idx[i]] += self.lr_p * adv * grad
        return float(np.mean(rets[-20:]))


def bench_mappo(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: MAPPO learns rendezvous coordination via
    centralized-value PPO."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = RendezvousEnv()
    agent = MAPPO(env)
    ret = agent.train(600, rng)
    out["synthetic_mappo_return"] = ret
    out["synthetic_mappo_learns"] = float(ret > 0)
    return out


if __name__ == "__main__":
    print(bench_mappo())
