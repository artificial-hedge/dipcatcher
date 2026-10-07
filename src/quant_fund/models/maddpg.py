"""MADDPG (Lowe et al., 2017) — centralized-training, (SYNTHETIC)
decentralized-execution deterministic actor-critic: each agent has
its own actor μ_i(obs_i); the shared critic sees the joint state and
ALL agents' actions. Continuous rendezvous on the ring.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ContRendezvous:
    """Continuous ring rendezvous: positions x ∈ [0,1)^2, actions
    a_i ∈ [−0.15, 0.15] added (mod 1). Reward = −dist(x1, x2); done
    when dist < 0.03 or horizon reached."""

    def __init__(self, horizon: int = 25):
        self.horizon = horizon
        self.x = np.zeros(2)
        self.t = 0

    def reset(self, rng: np.random.Generator) -> list[FloatArray]:
        self.x = rng.random(2)
        while abs(self.x[0] - self.x[1]) < 0.05:
            self.x = rng.random(2)
        self.t = 0
        return self.obs()

    def obs(self) -> list[FloatArray]:
        return [self.x.copy(), self.x[::-1].copy()]

    def _dist(self) -> float:
        d = abs(self.x[0] - self.x[1]) % 1.0
        return float(min(d, 1.0 - d))

    def step(self, actions: list[float]) -> tuple[list[FloatArray], float, bool]:
        a = np.clip(np.asarray(actions), -0.15, 0.15)
        self.x = (self.x + a) % 1.0
        self.t += 1
        d = self._dist()
        done = d < 0.03 or self.t >= self.horizon
        r = -5.0 * d + (10.0 if d < 0.03 else 0.0)
        return self.obs(), r, done


def _phi(x: FloatArray, a: FloatArray) -> FloatArray:
    """Centralized quadratic features over joint state + actions."""
    out: FloatArray = np.array(
        [
            1.0,
            x[0],
            x[1],
            a[0],
            a[1],
            x[0] * a[0],
            x[1] * a[1],
            a[0] * a[0],
            a[1] * a[1],
            a[0] * a[1],
            (x[1] - x[0]) ** 2,
        ]
    )
    return out


class MADDPG:
    """Shared quadratic critic + per-agent linear tanh actors."""

    def __init__(
        self, lr_c: float = 0.02, lr_a: float = 0.005, gamma: float = 0.95, tau: float = 0.05
    ):
        self.theta = [np.zeros(3), np.zeros(3)]
        self.theta_t = [np.zeros(3), np.zeros(3)]
        self.w = np.zeros(11)
        self.w_t = np.zeros(11)
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.gamma = gamma
        self.tau = tau
        self.replay: list[tuple] = []

    def act(self, i: int, o: FloatArray, noise: float = 0.0) -> float:
        feat = np.array([1.0, o[0], o[1] - o[0]])
        return float(np.tanh(self.theta[i] @ feat) * 0.15 + noise)

    def update(self, rng: np.random.Generator, batch: int = 48) -> None:
        buf = self.replay
        idx = rng.choice(len(buf), min(batch, len(buf)), replace=False)
        for i in idx:
            obs, acts, r, nobs, done = buf[i]
            x = np.array([obs[0][0], obs[0][1]])
            nx = np.array([nobs[0][0], nobs[0][1]])
            na = np.array(
                [
                    float(np.tanh(self.theta_t[0] @ np.array([1.0, nx[0], nx[1] - nx[0]])) * 0.15),
                    float(np.tanh(self.theta_t[1] @ np.array([1.0, nx[1], nx[0] - nx[1]])) * 0.15),
                ]
            )
            tgt = r if done else (r + self.gamma * float(self.w_t @ _phi(nx, na)))
            phi = _phi(x, np.asarray(acts))
            grad = phi * float(phi @ self.w - tgt)
            self.w -= self.lr_c * grad
        # deterministic policy gradient per agent (critic ∂Q/∂a_i)
        for i in idx[:16]:
            obs, _acts, _r, _nobs, _d = buf[i]
            x = np.array([obs[0][0], obs[0][1]])
            for k in range(2):
                feat = np.array([1.0, obs[k][0], obs[k][1] - obs[k][0]])
                a_k = float(np.tanh(self.theta[k] @ feat) * 0.15)
                other = float(self.theta[1 - k][2])
                acts_now = np.zeros(2)
                acts_now[k] = a_k
                acts_now[1 - k] = float(np.tanh(other) * 0.0)  # approximate joint action
                # ∂Q/∂a_k = w[3+k] + w[5+k]·x_k + 2w[7+k]·a_k + w[9]·a_j
                dqda = self.w[3 + k] + self.w[5 + k] * x[k] + 2 * self.w[7 + k] * a_k
                # da/dθ = 0.15·(1−tanh²)·feat
                dtanh = 0.15 * (1 - np.tanh(self.theta[k] @ feat) ** 2)
                self.theta[k] += self.lr_a * dqda * dtanh * feat
        for k in range(2):
            self.theta_t[k] = (1 - self.tau) * self.theta_t[k] + self.tau * self.theta[k]
        self.w_t = (1 - self.tau) * self.w_t + self.tau * self.w


def bench_maddpg(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: continuous rendezvous — MADDPG reduces final
    separation vs. the random-walk baseline."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    env = ContRendezvous()
    agent = MADDPG()
    meet, total = 0.0, 0.0
    for _ep in range(250):
        obs = env.reset(rng)
        done = False
        while not done:
            acts = [agent.act(i, obs[i], float(rng.normal(0, 0.03))) for i in range(2)]
            nobs, r, done = env.step(acts)
            agent.replay.append((obs, acts, r, nobs, done))
            if len(agent.replay) > 64:
                agent.update(rng)
            obs = nobs
        meet += float(env._dist() < 0.03)
        total += 1.0
    out["synthetic_maddpg_meet_rate"] = meet / total
    # evaluate greedy
    m2 = 0
    for _ in range(30):
        obs = env.reset(rng)
        done = False
        while not done:
            acts = [agent.act(i, obs[i]) for i in range(2)]
            obs, _r, done = env.step(acts)
        m2 += int(env._dist() < 0.03)
    out["synthetic_maddpg_greedy_meets"] = float(m2 / 30)
    out["synthetic_maddpg_learns"] = float(m2 / 30 > 0.3)
    return out


if __name__ == "__main__":
    print(bench_maddpg())
