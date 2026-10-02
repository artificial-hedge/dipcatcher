"""Tabular MDP canon: policy evaluation (Bellman linear solve), value
iteration, policy iteration (Howard), and Q-value extraction.
``bench_mdp_solvers`` builds a stochastic grid-world-ish MDP where the
optimal policy is known, and gates VI/PI recovering its value function
and greedy policy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check(p: FloatArray, r: FloatArray, gamma: float) -> None:
    if p.ndim != 3 or p.shape[0] != p.shape[2] or p.shape[1] != r.shape[1]:
        raise ValueError("p must be (S,A,S) and r (S,A)")
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0,1)")
    ssum = p.sum(axis=2)
    if np.max(np.abs(ssum - 1.0)) > 1e-6:
        raise ValueError("transition rows must sum to 1")


def policy_eval(p: FloatArray, r: FloatArray, pi: FloatArray, gamma: float = 0.99) -> FloatArray:
    """V^pi = (I - g P_pi)^-1 r_pi by exact linear solve."""
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    _check(p, r, gamma)
    p_pi = np.einsum("sa,sat->st", pi, p)
    r_pi = np.einsum("sa,sa->s", pi, r)
    return np.asarray(np.linalg.solve(np.eye(p.shape[0]) - gamma * p_pi, r_pi))


def value_iteration(
    p: FloatArray,
    r: FloatArray,
    gamma: float = 0.99,
    tol: float = 1e-10,
    it: int = 10000,
) -> tuple[FloatArray, FloatArray]:
    """Returns (V*, greedy policy one-hot)."""
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    _check(p, r, gamma)
    v = np.zeros(p.shape[0])
    for _ in range(it):
        q = r + gamma * np.einsum("sat,t->sa", p, v)
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    q = r + gamma * np.einsum("sat,t->sa", p, v)
    pi = np.zeros_like(q)
    pi[np.arange(q.shape[0]), q.argmax(axis=1)] = 1.0
    return v, pi


def policy_iteration(
    p: FloatArray,
    r: FloatArray,
    gamma: float = 0.99,
    it: int = 100,
) -> tuple[FloatArray, FloatArray]:
    """Howard PI: exact policy eval + greedy improvement."""
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    _check(p, r, gamma)
    n_s, n_a = r.shape
    pi = np.full((n_s, n_a), 1.0 / n_a)
    for _ in range(it):
        v = policy_eval(p, r, pi, gamma)
        q = r + gamma * np.einsum("sat,t->sa", p, v)
        new = np.zeros_like(pi)
        new[np.arange(n_s), q.argmax(axis=1)] = 1.0
        if np.allclose(new, pi):
            pi = new
            break
        pi = new
    return v, pi


def q_values(p: FloatArray, r: FloatArray, v: FloatArray, gamma: float = 0.99) -> FloatArray:
    return np.asarray(
        np.asarray(r, dtype=np.float64)
        + gamma * np.einsum("sat,t->sa", np.asarray(p), np.asarray(v)),
        dtype=np.float64,
    )


def bench_mdp_solvers(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_s, n_a = 12, 3
    # random stochastic MDP with planted "go right" optimal structure
    p = rng.dirichlet(np.ones(n_s) * 0.4, size=(n_s, n_a))
    r = rng.standard_normal((n_s, n_a)) * 0.5
    # make action 2 uniformly rewarding toward an absorbing goal state
    p[:, 2, :] = 0.0
    p[:, 2, n_s - 1] = 1.0
    p[n_s - 1, :, :] = 0.0
    p[n_s - 1, :, n_s - 1] = 1.0
    r[:, 2] += 1.0
    r[n_s - 1, :] = 2.0
    gamma = 0.95
    v_vi, pi_vi = value_iteration(p, r, gamma)
    v_pi, pi_pi = policy_iteration(p, r, gamma)
    agree = float(np.mean(pi_vi.argmax(1) == pi_pi.argmax(1)))
    return {
        "synthetic_vi_pi_gap": float(np.max(np.abs(v_vi - v_pi))),
        "synthetic_policy_agree": agree,
        "synthetic_v_star_goal": float(v_vi[n_s - 1]),
        "synthetic_q_gap": float(np.max(np.abs(q_values(p, r, v_vi, gamma).max(axis=1) - v_vi))),
    }
