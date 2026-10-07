"""DDPG (Lillicrap et al., 2015) — deterministic actor-critic with a (SYNTHETIC)
quadratic-feature critic. Target networks + experience replay on a
1-D regulation task; policy gradient ∂Q/∂a·∂a/∂θ through the critic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def q_features(s: float, a: float) -> FloatArray:
    """Quadratic critic features [1, s, a, sa, s², a²]."""
    out: FloatArray = np.array([1.0, s, a, s * a, s * s, a * a])
    return out


class DDPG:
    """Deterministic policy μ(s) = clip(θ0 + θ1 s) with critic
    Q(s,a) = wᵀφ(s,a). Off-policy TD + deterministic policy gradient."""

    def __init__(self, a_lim: float = 2.0, gamma: float = 0.95, tau: float = 0.05):
        self.theta = np.zeros(2)
        self.w = np.zeros(6)
        self.theta_t = np.zeros(2)
        self.w_t = np.zeros(6)
        self.a_lim = a_lim
        self.gamma = gamma
        self.tau = tau
        self.replay: list[tuple[float, float, float, float]] = []

    def act(self, s: float, noise: float = 0.0) -> float:
        return float(np.clip(self.theta[0] + self.theta[1] * s + noise, -self.a_lim, self.a_lim))

    def q(self, s: float, a: float, target: bool = False) -> float:
        w = self.w_t if target else self.w
        return float(w @ q_features(s, a))

    def store(self, s: float, a: float, r: float, sp: float) -> None:
        self.replay.append((s, a, r, sp))

    def update(
        self, rng: np.random.Generator, batch: int = 32, lr_c: float = 0.05, lr_a: float = 0.02
    ) -> None:
        buf = np.array(self.replay)
        idx = rng.choice(len(buf), size=min(batch, len(buf)), replace=False)
        x_rows, y = [], []
        for i in idx:
            s, a, r, sp = buf[i]
            ap = float(self.theta_t[0] + self.theta_t[1] * sp)
            ap = float(np.clip(ap, -self.a_lim, self.a_lim))
            tgt = r + self.gamma * self.q(sp, ap, target=True)
            x_rows.append(q_features(s, a))
            y.append(tgt)
        # ridge critic update (batch least squares step)
        x_m = np.array(x_rows)
        grad_c = x_m.T @ (x_m @ self.w - np.array(y)) / len(idx)
        self.w = self.w - lr_c * grad_c / (np.abs(grad_c).max() + 1e-9)
        # actor: ascend E_s[Q(s, μ(s))] via chain rule
        ga = np.zeros(2)
        for i in idx:
            s = float(buf[i][0])
            a = self.act(s)
            dqda = self.w[2] + self.w[3] * s + 2 * self.w[5] * a
            ga += dqda * np.array([1.0, s])
        self.theta = self.theta + lr_a * ga / len(idx)
        # soft target updates
        self.w_t = (1 - self.tau) * self.w_t + self.tau * self.w
        self.theta_t = (1 - self.tau) * self.theta_t + self.tau * self.theta


def bench_ddpg(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: regulation task x'=x+a, r=−x²; DDPG learns μ(s)≈−s
    → cost decreases vs. the zero policy."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    agent = DDPG()
    # collect + learn interleaved
    s = float(rng.uniform(-1, 1))
    for _t in range(2_000):
        a = agent.act(s, noise=float(rng.normal(0, 0.3)))
        sp = s + a
        r = -(s * s)
        agent.store(s, a, r, sp)
        if len(agent.replay) > 64:
            agent.update(rng)
        s = sp if abs(sp) < 3 else float(rng.uniform(-1, 1))
    # evaluate: greedy rollout
    s = 0.8
    for _ in range(10):
        s = s + agent.act(s)
    out["synthetic_ddpg_final_state"] = abs(s)
    out["synthetic_ddpg_policy_slope"] = float(-agent.theta[1])
    out["synthetic_ddpg_learns"] = float(abs(s) < 0.2)
    return out


if __name__ == "__main__":
    print(bench_ddpg())
