"""TD3 (Fujimoto et al., 2018) — twin delayed deep deterministic (SYNTHETIC)
policy gradient: twin critics with min-target, delayed policy
updates, target-policy smoothing, and soft target networks. Same
quadratic-critic skeleton as `ddpg` to isolate the algorithmic delta.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.ddpg import q_features

FloatArray = NDArray[np.float64]


class TD3:
    """Twin-critic deterministic actor-critic."""

    def __init__(
        self,
        a_lim: float = 2.0,
        gamma: float = 0.95,
        tau: float = 0.05,
        policy_delay: int = 2,
        smooth_noise: float = 0.2,
    ):
        self.theta = np.zeros(2)
        self.w1 = np.zeros(6)
        self.w2 = np.zeros(6)
        self.theta_t = np.zeros(2)
        self.w1_t = np.zeros(6)
        self.w2_t = np.zeros(6)
        self.a_lim = a_lim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.smooth_noise = smooth_noise
        self.replay: list[tuple[float, float, float, float]] = []
        self.it = 0

    def act(self, s: float, noise: float = 0.0) -> float:
        return float(np.clip(self.theta[0] + self.theta[1] * s + noise, -self.a_lim, self.a_lim))

    def store(self, s: float, a: float, r: float, sp: float) -> None:
        self.replay.append((s, a, r, sp))

    def update(
        self, rng: np.random.Generator, batch: int = 32, lr_c: float = 0.05, lr_a: float = 0.02
    ) -> None:
        self.it += 1
        buf = np.array(self.replay)
        idx = rng.choice(len(buf), size=min(batch, len(buf)), replace=False)
        x_rows, y = [], []
        for i in idx:
            s, a, r, sp = buf[i]
            ap = float(self.theta_t[0] + self.theta_t[1] * sp)
            ap += float(rng.normal(0, self.smooth_noise))  # smoothing
            ap = float(np.clip(ap, -self.a_lim, self.a_lim))
            q1 = float(self.w1_t @ q_features(sp, ap))
            q2 = float(self.w2_t @ q_features(sp, ap))
            y.append(r + self.gamma * min(q1, q2))  # twin min target
            x_rows.append(q_features(s, a))
        x_m = np.array(x_rows)
        err = np.array(y)
        for w in (self.w1, self.w2):
            g = x_m.T @ (x_m @ w - err) / len(idx)
            w -= lr_c * g / (np.abs(g).max() + 1e-9)
        # delayed policy update
        if self.it % self.policy_delay == 0:
            ga = np.zeros(2)
            for i in idx:
                s = float(buf[i][0])
                a = float(np.clip(self.theta[0] + self.theta[1] * s, -self.a_lim, self.a_lim))
                dqda = self.w1[2] + self.w1[3] * s + 2 * self.w1[5] * a
                ga += dqda * np.array([1.0, s])
            self.theta += lr_a * ga / len(idx)
            self.theta_t = (1 - self.tau) * self.theta_t + self.tau * self.theta
            self.w1_t = (1 - self.tau) * self.w1_t + self.tau * self.w1
            self.w2_t = (1 - self.tau) * self.w2_t + self.tau * self.w2


def bench_td3(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: same regulation task as DDPG — TD3 converges to
    μ(s)≈−s and its twin critics agree more than a single critic
    overestimates."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    agent = TD3()
    s = float(rng.uniform(-1, 1))
    for _ in range(2_000):
        a = agent.act(s, noise=float(rng.normal(0, 0.3)))
        sp = s + a
        agent.store(s, a, -(s * s), sp)
        if len(agent.replay) > 64:
            agent.update(rng)
        s = sp if abs(sp) < 3 else float(rng.uniform(-1, 1))
    s = 0.8
    for _ in range(10):
        s = s + agent.act(s)
    out["synthetic_td3_final_state"] = abs(s)
    out["synthetic_td3_policy_slope"] = float(-agent.theta[1])
    out["synthetic_td3_critic_gap"] = float(np.abs(agent.w1 - agent.w2).max())
    out["synthetic_td3_learns"] = float(abs(s) < 0.2)
    return out


if __name__ == "__main__":
    print(bench_td3())
