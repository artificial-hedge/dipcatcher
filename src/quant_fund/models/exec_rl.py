"""Reinforcement-learning optimal execution with auction logic (Exec-Summary
Feature 2). Simulated market: GBM mid with short-horizon autocorrelation plus
temporary impact, a limit step and a terminal closing auction. A tabular
Q-learning agent learns order-placement schedules to minimize inventory-
penalized implementation shortfall vs TWAP.

Synthetic bench: Q-agent vs TWAP shortfall on held-out paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

_ACTIONS = np.array([0.0, 0.1, 0.25, 0.5, 1.0])


@dataclass
class MarketSim:
    """GBM mid with AR(1) flow pressure and square-root temporary impact."""

    s0: float = 100.0
    sigma: float = 0.02
    impact: float = 0.08
    rho: float = 0.3
    steps: int = 10
    auction_penalty: float = 0.002

    def episode(self, rng: np.random.Generator) -> FloatArray:
        """Returns (steps+2,) mid path incl. auction print."""
        eps = rng.standard_normal(self.steps + 1)
        flow = np.zeros(self.steps + 1)
        for t in range(1, self.steps + 1):
            flow[t] = self.rho * flow[t - 1] + eps[t]
        mid = np.empty(self.steps + 2)
        mid[0] = self.s0
        for t in range(1, self.steps + 1):
            mid[t] = mid[t - 1] * np.exp(self.sigma * flow[t] / np.sqrt(self.steps))
        mid[self.steps + 1] = mid[self.steps] * (1.0 - self.auction_penalty)
        return mid


@dataclass
class ExecState:
    t: int
    remaining: float
    mid: float


def _exec_price(mid: float, qty_frac: float, sim: MarketSim) -> float:
    return float(mid * (1.0 - sim.impact * np.sqrt(max(qty_frac, 0.0))))


def run_episode(
    policy, mid: FloatArray, sim: MarketSim, inventory: float = 1.0
) -> dict[str, float]:
    """Sell `inventory` units over sim.steps slots + auction."""
    remaining = inventory
    proceeds = 0.0
    rem_hist = []
    for t in range(sim.steps):
        frac = float(policy.act(ExecState(t, remaining, mid[t])))
        qty = remaining * float(np.clip(frac, 0.0, 1.0))
        proceeds += qty * _exec_price(mid[t], qty / inventory, sim)
        remaining -= qty
        rem_hist.append(remaining)
        policy.observe(mid[t + 1] / mid[t] - 1.0)
    # closing auction: dump remainder at auction print
    proceeds += remaining * mid[sim.steps + 1]
    is_bps = 1e4 * (proceeds / inventory / mid[0] - 1.0)
    inv_var = float(np.var(rem_hist)) if rem_hist else 0.0
    return {"shortfall_bps": is_bps, "inventory_var": inv_var}


class TwapPolicy:
    def __init__(self, steps: int):
        self.steps = steps

    def act(self, s: ExecState) -> float:
        return 1.0 / max(self.steps - s.t, 1)

    def observe(self, ret: float) -> None:
        return None


class QExecAgent:
    """Tabular Q-learner on (time bucket, remaining bucket, momentum sign)."""

    def __init__(
        self,
        steps: int,
        t_bins: int | None = None,
        q_bins: int = 5,
        alpha: float = 0.4,
        gamma: float = 0.97,
        eps: float = 0.15,
        lam: float = 0.5,
    ):
        self.steps = steps
        self.t_bins = t_bins or steps
        self.q_bins = q_bins
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.lam = lam
        self.q = np.zeros((self.t_bins, q_bins, 2, len(_ACTIONS)))
        self._s: tuple[int, int, int] | None = None
        self._a = 0
        self._reward = 0.0

    def _state(self, s: ExecState) -> tuple[int, int, int]:
        tb = min(int(s.t / self.steps * self.t_bins), self.t_bins - 1)
        qb = min(int(s.remaining * self.q_bins), self.q_bins - 1)
        mb = int(getattr(self, "_mom", 0.0) > 0.0)
        return (tb, qb, mb)

    def act(self, s: ExecState) -> float:
        st = self._state(s)
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(len(_ACTIONS)))
        else:
            a = int(np.argmax(self.q[st]))
        if self._s is not None:
            self._update(st)
        self._s, self._a = st, a
        self._reward = 0.0
        return float(_ACTIONS[a])

    def observe(self, ret: float) -> None:
        self._mom = ret
        self._reward += -abs(ret) * self.lam

    def feed_proceeds(self, r: float) -> None:
        self._reward += r

    def _update(self, s_next: tuple[int, int, int]) -> None:
        if not (self._s is not None):
            raise ValueError("self._s is not None")
        t_, q_, m_ = self._s
        td = (
            self._reward
            + self.gamma * float(np.max(self.q[s_next]))
            - float(self.q[t_, q_, m_, self._a])
        )
        self.q[t_, q_, m_, self._a] += self.alpha * td

    def finish(self, terminal_r: float) -> None:
        if not (self._s is not None):
            raise ValueError("self._s is not None")
        self._reward += terminal_r
        t_, q_, m_ = self._s
        td = self._reward - float(self.q[t_, q_, m_, self._a])
        self.q[t_, q_, m_, self._a] += self.alpha * td
        self._s = None

    rng = np.random.default_rng(0)


def train_q_agent(
    sim: MarketSim, episodes: int, rng: np.random.Generator, inventory: float = 1.0
) -> QExecAgent:
    agent = QExecAgent(sim.steps)
    agent.rng = rng
    for _ in range(episodes):
        mid = sim.episode(rng)
        remaining = inventory
        agent._mom = 0.0
        for t in range(sim.steps):
            frac = agent.act(ExecState(t, remaining, mid[t]))
            qty = remaining * float(np.clip(frac, 0.0, 1.0))
            px = _exec_price(mid[t], qty / inventory, sim)
            agent.feed_proceeds(qty * px / (inventory * mid[0]) - qty / inventory)
            remaining -= qty
            agent.observe(mid[t + 1] / mid[t] - 1.0)
        agent.finish(remaining * mid[sim.steps + 1] / (inventory * mid[0]))
    agent.eps = 0.0
    return agent


def bench_exec_rl(seed: int = 7) -> dict[str, float]:
    sim = MarketSim(steps=8)
    rng = np.random.default_rng(seed)
    agent = train_q_agent(sim, 300, rng)
    twap = TwapPolicy(sim.steps)
    rng_eval = np.random.default_rng(seed + 1)
    dqn_is, twap_is, dqn_var, twap_var = [], [], [], []
    for _ in range(150):
        mid = sim.episode(rng_eval)
        r1 = run_episode(agent, mid, sim)
        r2 = run_episode(twap, mid, sim)
        dqn_is.append(r1["shortfall_bps"])
        twap_is.append(r2["shortfall_bps"])
        dqn_var.append(r1["inventory_var"])
        twap_var.append(r2["inventory_var"])
    d, t = float(np.mean(dqn_is)), float(np.mean(twap_is))
    return {
        "synthetic_exec_rl_agent_is_bps": d,
        "synthetic_exec_rl_twap_is_bps": t,
        "synthetic_exec_rl_margin_bps": d - t,
        "synthetic_exec_rl_agent_inv_var": float(np.mean(dqn_var)),
        "synthetic_exec_rl_twap_inv_var": float(np.mean(twap_var)),
        "synthetic_exec_rl_agent_beats_twap_rate": float(
            np.mean(np.array(dqn_is) > np.array(twap_is))
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_exec_rl(), indent=1))
