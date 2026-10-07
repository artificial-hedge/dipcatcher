"""COMA (Foerster et al., 2018) — counterfactual multi-agent policy (SYNTHETIC)
gradients: centralized critic Q(s, a_joint), per-agent advantage
A_i = Q(s, a) − Σ_{a_i'} π_i(a_i')·Q(s, a_{−i}, a_i'), removing the
agent's own action from the credit signal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.vdn import RendezvousEnv, obs_index

FloatArray = NDArray[np.float64]


def _softmax(z: FloatArray) -> FloatArray:
    e = np.exp(z - z.max())
    out: FloatArray = np.asarray(e / e.sum())
    return out


class COMA:
    """Tabular centralized critic Q[glob_state][a0][a1] + per-agent
    softmax policies on local obs."""

    def __init__(
        self, env: RendezvousEnv, lr_c: float = 0.1, lr_a: float = 0.05, gamma: float = 0.95
    ):
        self.env = env
        self.gamma = gamma
        self.lr_c = lr_c
        self.lr_a = lr_a
        n_gs = env.n_pos**env.n_agents
        n_ls = env.n_pos * env.n_pos
        self.critic = np.zeros((n_gs, env.n_actions, env.n_actions))
        self.theta = [np.zeros((n_ls, env.n_actions)) for _ in range(2)]

    def _pi(self, i: int, o_idx: int) -> FloatArray:
        return _softmax(self.theta[i][o_idx])

    def _glob(self, obs: list[FloatArray]) -> int:
        return obs_index(obs[0], self.env.n_pos)

    def counterfactual_adv(self, sg: int, o_idx: list[int], acts: list[int], i: int) -> float:
        """A_i = Q(s,a) − Σ_{a'} π_i(a'|o_i) Q(s, (a_{−i}, a'))."""
        pi = self._pi(i, o_idx[i])
        base = 0.0
        for ap in range(self.env.n_actions):
            joint = [acts[0], acts[1]]
            joint[i] = ap
            base += float(pi[ap]) * self.critic[sg, joint[0], joint[1]]
        return float(self.critic[sg, acts[0], acts[1]] - base)

    def train(self, episodes: int, rng: np.random.Generator, eps: float = 0.1) -> float:
        rets = []
        for _ in range(episodes):
            obs = self.env.reset(rng)
            ret, done = 0.0, False
            while not done:
                o_idx = [obs_index(o, self.env.n_pos) for o in obs]
                pis = [self._pi(i, o_idx[i]) for i in range(2)]
                acts = []
                for i in range(2):
                    if rng.random() < eps:
                        acts.append(int(rng.integers(3)))
                    else:
                        acts.append(int(rng.choice(3, p=pis[i])))
                nobs, r, done = self.env.step(acts)
                ret += r
                sg = self._glob(obs)
                nsg = self._glob(nobs)
                # critic TD: Q(sg, a0, a1) ← r + γ E_{a'~π} Q(nsg, ·)
                if done:
                    tgt = r
                else:
                    no_idx = [obs_index(o, self.env.n_pos) for o in nobs]
                    npi = [self._pi(i, no_idx[i]) for i in range(2)]
                    ev = sum(
                        npi[0][a0] * npi[1][a1] * self.critic[nsg, a0, a1]
                        for a0 in range(3)
                        for a1 in range(3)
                    )
                    tgt = r + self.gamma * float(ev)
                self.critic[sg, acts[0], acts[1]] += self.lr_c * (
                    tgt - self.critic[sg, acts[0], acts[1]]
                )
                # counterfactual policy gradient per agent
                for i in range(2):
                    adv = self.counterfactual_adv(sg, o_idx, acts, i)
                    grad = -pis[i]
                    grad[acts[i]] += 1.0
                    self.theta[i][o_idx[i]] += self.lr_a * adv * grad
                obs = nobs
            rets.append(ret)
        return float(np.mean(rets[-20:]))


def bench_coma(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: COMA learns coordination on rendezvous with
    counterfactual credit; return beats the uncoordinated baseline."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = RendezvousEnv()
    agent = COMA(env)
    ret = agent.train(800, rng)
    out["synthetic_coma_return"] = ret
    out["synthetic_coma_learns"] = float(ret > 0)
    # counterfactual baseline reduces variance: adv is 0 when the
    # agent's action doesn't matter (critic column constant)
    sg = 0
    agent.critic[sg, :, :] = 3.0  # flat → zero advantage
    adv = agent.counterfactual_adv(sg, [0, 0], [1, 1], 0)
    out["synthetic_coma_flat_adv"] = abs(adv)
    return out


if __name__ == "__main__":
    print(bench_coma())
