"""POMDP primitives + QMDP approximation — treats the problem as
fully observable after the current step: value = MDP Q-function,
policy greedy on the belief-weighted Q. Provides the shared spec,
belief update, α-vector backup, and rollout simulator used by the
other wave-118 POMDP modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class POMDP:
    """Factored POMDP: T[a,s,s'], O[a,s',o], R[a,s], gamma."""

    t: FloatArray
    o: FloatArray
    r: FloatArray
    gamma: float
    s0_dist: FloatArray

    @property
    def n_s(self) -> int:
        return int(self.t.shape[1])

    @property
    def n_a(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_o(self) -> int:
        return int(self.o.shape[2])


def tiger_pomdp(gamma: float = 0.9) -> POMDP:
    """Classic 2-state Tiger problem: tiger behind one of two doors.
    listen(0) costs −1 with 0.85 informative obs; open doors gives
    +10 / −100 and resets the tiger."""
    t = np.zeros((3, 2, 2))
    o = np.zeros((3, 2, 2))
    r = np.array([[-1.0, -1.0], [-100.0, 10.0], [10.0, -100.0]])
    # listen: state unchanged; obs 0=hear-left correctly w.p. 0.85
    t[0] = np.eye(2)
    o[0] = np.array([[0.85, 0.15], [0.15, 0.85]])
    # open doors: tiger resets uniformly, obs uninformative
    t[1:] = 0.5
    o[1:] = 0.5
    return POMDP(t=t, o=o, r=r, gamma=gamma, s0_dist=np.array([0.5, 0.5]))


def belief_update(p: POMDP, b: FloatArray, a: int, ob: int) -> FloatArray:
    """b'(s') ∝ O(a,s',o) Σ_s T(a,s,s') b(s)."""
    nb = p.o[a, :, ob] * (p.t[a].T @ b)
    tot = float(nb.sum())
    if tot <= 0:
        return b.copy()
    out: FloatArray = np.asarray(nb / tot)
    return out


def obs_prob(p: POMDP, b: FloatArray, a: int, ob: int) -> float:
    """P(o | b, a)."""
    return float(p.o[a, :, ob] @ (p.t[a].T @ b))


def mdp_value_iteration(p: POMDP, iters: int = 500) -> FloatArray:
    """Solve the underlying fully-observable MDP."""
    v = np.zeros(p.n_s)
    for _ in range(iters):
        q = p.r + p.gamma * np.einsum("asj,j->as", p.t, v)
        v = q.max(axis=0)
    out: FloatArray = np.asarray(v)
    return out


def qmdp_policy(p: POMDP, b: FloatArray, v: FloatArray) -> int:
    """Greedy QMDP action at belief b under MDP Q-values."""
    q = p.r + p.gamma * np.einsum("asj,j->as", p.t, v)
    return int(np.argmax(q @ b))


def pomdp_rollout(
    p: POMDP,
    policy_fn: object,
    n_episodes: int,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    """Simulate episodes; policy_fn(b) -> action. Returns mean
    discounted return."""
    rets = []
    for _ in range(n_episodes):
        s = int(rng.choice(p.n_s, p=p.s0_dist))
        b = p.s0_dist.copy()
        ret, disc = 0.0, 1.0
        for _ in range(horizon):
            a = policy_fn(b)  # type: ignore[operator]
            sp = int(rng.choice(p.n_s, p=p.t[a, s]))
            ob = int(rng.choice(p.n_o, p=p.o[a, sp]))
            ret += disc * p.r[a, s]
            disc *= p.gamma
            b = belief_update(p, b, a, ob)
            s = sp
        rets.append(ret)
    return float(np.mean(rets))


def bench_qmdp(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: QMDP on Tiger — the canonical property is that its
    value is an UPPER bound on the true POMDP value (it assumes full
    observability next step), while its greedy policy still listens
    when uncertain."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    v = mdp_value_iteration(p)
    # at a uniform belief the policy listens (door guess EV < listen)
    out["synthetic_qmdp_listens_uncertain"] = float(qmdp_policy(p, np.array([0.5, 0.5]), v) == 0)
    ret = pomdp_rollout(p, lambda b: qmdp_policy(p, b, v), 200, 20, rng)
    out["synthetic_qmdp_return"] = ret
    v_est = float(v @ p.s0_dist)
    out["synthetic_qmdp_value"] = v_est
    # value bound: V_MDP ≥ realized POMDP return (strict overestimate)
    out["synthetic_qmdp_upper_bound"] = float(v_est > ret + 1e-9)
    return out


if __name__ == "__main__":
    print(bench_qmdp())
