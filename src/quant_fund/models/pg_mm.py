"""Policy-gradient market maker (Exec-Summary PG-MM item). REINFORCE
agent quotes bid/ask offsets in a Poisson-fill LOB simulator
(fill intensity A*exp(-k*offset)) with quadratic inventory penalty.

Synthetic bench: learned Gaussian-offset policy vs naive symmetric
quoting — terminal wealth gap and inventory variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray


@dataclass
class LobEnv:
    """Avellaneda-Stoikov-style fills: Poisson rate A*exp(-k*offset)."""

    s0: float = 100.0
    sigma: float = 0.5
    a_fill: float = 1.5
    k_fill: float = 1.5
    steps: int = 40
    lam: float = 0.02
    q_max: int = 5

    def episode(self, policy, rng: np.random.Generator) -> dict[str, Any]:
        mid = self.s0
        cash = 0.0
        q = 0
        q_hist = []
        feats_hist = []
        act_hist = []
        rewards = []
        prev_mid = mid
        for t in range(self.steps):
            mid *= np.exp(self.sigma / np.sqrt(self.steps) * rng.standard_normal())
            feat = np.array([1.0, q / self.q_max, t / self.steps])
            off = policy.act(feat, rng)
            d_bid, d_ask = float(np.clip(off[0], 0.05, 3.0)), float(np.clip(off[1], 0.05, 3.0))
            flow = 0.0
            for side, d in (("bid", d_bid), ("ask", d_ask)):
                n = rng.poisson(self.a_fill * np.exp(-self.k_fill * d))
                n = min(n, self.q_max - q if side == "bid" else self.q_max + q)
                if side == "bid" and n > 0:
                    cash -= n * (mid - d)
                    flow += n * d
                    q += n
                elif side == "ask" and n > 0:
                    cash += n * (mid + d)
                    flow += n * d
                    q -= n
            # per-step reward: spread capture + inventory mark - penalty
            r_t = flow + q * (mid - prev_mid) - self.lam * q * q * mid * 0.02
            rewards.append(r_t)
            prev_mid = mid
            q_hist.append(q)
            feats_hist.append(feat)
            act_hist.append([d_bid, d_ask])
        wealth = cash + q * mid - self.lam * q * q * mid
        return {
            "wealth": wealth,
            "inv_var": float(np.var(q_hist)),
            "feats": feats_hist,
            "acts": act_hist,
            "rewards": rewards,
        }


class NaiveMM:
    def act(self, feat: FloatArray, rng: np.random.Generator) -> FloatArray:
        return np.array([0.5, 0.5])


class GaussianPolicy:
    """Linear-Gaussian offset policy: mean = W @ feat, fixed sigma."""

    def __init__(self, n_feat: int = 3, sigma: float = 0.4):
        self.w = np.full((2, n_feat), 0.5 / 1.0)
        self.w[:, 0] = 0.5
        self.sigma = sigma

    def act(self, feat: FloatArray, rng: np.random.Generator) -> FloatArray:
        return np.asarray(self.w @ feat + self.sigma * rng.standard_normal(2))

    def grad_logp(self, feat: FloatArray, act: FloatArray) -> FloatArray:
        """d log pi / dW for each dim: (a - mu) * feat / sigma^2."""
        mu = self.w @ feat
        return np.outer((act - mu) / self.sigma**2, feat)


def train_pg_mm(
    env: LobEnv, epochs: int, rng: np.random.Generator, lr: float = 0.004
) -> GaussianPolicy:
    pol = GaussianPolicy()
    adv_mu, adv_var = 0.0, 1.0
    gamma = 0.95
    for _ in range(epochs):
        out = env.episode(pol, rng)
        rtg = np.zeros(env.steps)
        acc = 0.0
        for t in range(env.steps - 1, -1, -1):
            acc = out["rewards"][t] + gamma * acc
            rtg[t] = acc
        m = float(rtg.mean())
        adv_mu = 0.95 * adv_mu + 0.05 * m
        adv_var = 0.95 * adv_var + 0.05 * float(rtg.var())
        grad = np.zeros_like(pol.w)
        scale = max(np.sqrt(adv_var), 1e-6)
        for t, (feat, act) in enumerate(zip(out["feats"], out["acts"], strict=True)):
            adv = (rtg[t] - adv_mu) / scale
            grad += adv * pol.grad_logp(feat, np.array(act))
        pol.w += lr * grad / env.steps
        pol.w = np.clip(pol.w, -3.0, 3.0)
    return pol


def bench_pg_mm(seed: int = 7) -> dict[str, float]:
    env = LobEnv()
    rng = np.random.default_rng(seed)
    pol = train_pg_mm(env, 400, rng)
    rng_eval = np.random.default_rng(seed + 1)
    naive = NaiveMM()
    lw, nw, lv, nv = [], [], [], []
    for _ in range(150):
        r1 = env.episode(pol, rng_eval)
        r2 = env.episode(naive, rng_eval)
        lw.append(r1["wealth"])
        nw.append(r2["wealth"])
        lv.append(r1["inv_var"])
        nv.append(r2["inv_var"])
    lw_a, nw_a = np.array(lw), np.array(nw)
    return {
        "synthetic_pg_mm_learned_wealth": float(lw_a.mean()),
        "synthetic_pg_mm_naive_wealth": float(nw_a.mean()),
        "synthetic_pg_mm_wealth_margin": float(lw_a.mean() - nw_a.mean()),
        "synthetic_pg_mm_learned_inv_var": float(np.mean(lv)),
        "synthetic_pg_mm_naive_inv_var": float(np.mean(nv)),
        "synthetic_pg_mm_beats_naive_rate": float(np.mean(lw_a > nw_a)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_pg_mm(), indent=1))
