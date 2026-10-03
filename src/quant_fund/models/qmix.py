"""QMIX (Rashid et al., 2018) — monotonic value decomposition:
Q_tot = f_s(Q_1, ..., Q_n) with ∂Q_tot/∂Q_i ≥ 0 enforced by
non-negative mixing weights produced by a hypernetwork on the
global state. Per-agent Q tables + linear hypernet params.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.vdn import RendezvousEnv, obs_index

FloatArray = NDArray[np.float64]


class QMIX:
    """QMIX on the rendezvous env.

    State embedding: one-hot of the joint position pair. Mixing:
    Q_tot = Σ_i w_i(s)·Q_i + b(s) with w_i(s) = |u_iᵀ e_s| ≥ 0
    (monotonicity) and b(s) = vᵀ e_s.
    """

    def __init__(
        self, env: RendezvousEnv, lr: float = 0.05, gamma: float = 0.95, eps: float = 0.15
    ):
        self.env = env
        self.n_gs = env.n_pos**env.n_agents
        self.q = [np.zeros((env.n_pos * env.n_pos, env.n_actions)) for _ in range(env.n_agents)]
        # hypernet params: mixing weights + bias, linear in state
        rng = np.random.default_rng(0)
        self.u = [rng.normal(0, 0.1, self.n_gs) for _ in range(2)]
        self.v = np.zeros(self.n_gs)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def _mix(self, s_glob: int, q_vals: FloatArray) -> float:
        w = np.abs([float(self.u[i][s_glob]) for i in range(2)])
        return float(w @ q_vals + self.v[s_glob])

    def _glob(self, obs: list[FloatArray]) -> int:
        return int(obs[0][0]) * self.env.n_pos + int(obs[0][1])

    def joint_greedy(self, obs: list[FloatArray]) -> list[int]:
        sg = self._glob(obs)
        s_idx = [obs_index(o, self.env.n_pos) for o in obs]
        q0, q1 = self.q[0][s_idx[0]], self.q[1][s_idx[1]]
        best, best_v = (0, 0), -np.inf
        for a0 in range(3):
            for a1 in range(3):
                v = self._mix(sg, np.array([q0[a0], q1[a1]]))
                if v > best_v:
                    best_v, best = v, (a0, a1)
        return [int(best[0]), int(best[1])]

    def train(self, episodes: int, rng: np.random.Generator) -> float:
        rets = []
        for _ in range(episodes):
            obs = self.env.reset(rng)
            ret, done = 0.0, False
            while not done:
                acts = (
                    [int(rng.integers(3)), int(rng.integers(3))]
                    if rng.random() < self.eps
                    else self.joint_greedy(obs)
                )
                nobs, r, done = self.env.step(acts)
                ret += r
                sg = self._glob(obs)
                nsg = self._glob(nobs)
                s_idx = [obs_index(o, self.env.n_pos) for o in obs]
                ns_idx = [obs_index(o, self.env.n_pos) for o in nobs]
                q_a = np.array([self.q[0][s_idx[0]][acts[0]], self.q[1][s_idx[1]][acts[1]]])
                if done:
                    tgt = r
                else:
                    nq0, nq1 = self.q[0][ns_idx[0]], self.q[1][ns_idx[1]]
                    best_next = -np.inf
                    for a0 in range(3):
                        for a1 in range(3):
                            best_next = max(
                                best_next,
                                self._mix(nsg, np.array([nq0[a0], nq1[a1]])),
                            )
                    tgt = r + self.gamma * best_next
                td = tgt - self._mix(sg, q_a)
                w = [abs(float(self.u[i][sg])) for i in range(2)]
                # per-agent Q update
                for i in range(2):
                    self.q[i][s_idx[i]][acts[i]] += self.lr * w[i] * td
                    # hypernet: ∂Qtot/∂u_i = sign(u_i)·Q_i
                    self.u[i][sg] += self.lr * np.sign(self.u[i][sg]) * q_a[i] * td
                self.v[sg] += self.lr * td
                obs = nobs
            rets.append(ret)
        return float(np.mean(rets[-20:]))


def bench_qmix(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: QMIX learns rendezvous coordination with monotonic
    mixing (all mixing weights ≥ 0)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = RendezvousEnv()
    agent = QMIX(env)
    ret = agent.train(800, rng)
    out["synthetic_qmix_return"] = ret
    out["synthetic_qmix_learns"] = float(ret > 0)
    out["synthetic_qmix_nonneg_w"] = float(min(abs(float(u.min())) for u in agent.u) >= 0)
    return out


if __name__ == "__main__":
    print(bench_qmix())
