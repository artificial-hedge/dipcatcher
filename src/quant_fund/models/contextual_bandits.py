"""Contextual bandit canon: disjoint LinUCB (Li, Chu, (SYNTHETIC)
Langford & Schapire 2010) and linear Thompson sampling
(Agrawal & Goyal 2013) on a synthetic linear-reward
environment, against an epsilon-greedy linear baseline.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _env(rng: np.random.Generator, k: int, d: int, steps: int) -> tuple[FloatArray, FloatArray]:
    thetas = rng.standard_normal((k, d))
    ctx = rng.standard_normal((steps, d))
    return thetas, ctx


def _oracle_regret(thetas: FloatArray, ctx: FloatArray, picks: list[int]) -> float:
    ideal = (ctx @ thetas.T).max(axis=1)
    got = np.array([ctx[t] @ thetas[a] for t, a in enumerate(picks)])
    return float(np.sum(ideal - got))


def linucb(
    thetas: FloatArray,
    ctx: FloatArray,
    rng: np.random.Generator,
    alpha: float = 0.6,
    noise: float = 0.25,
) -> float:
    """Disjoint LinUCB; returns pseudo-regret."""
    k, d = thetas.shape
    a_mats = np.stack([np.eye(d) for _ in range(k)])
    b_vecs = np.zeros((k, d))
    picks: list[int] = []
    for t in range(ctx.shape[0]):
        x = ctx[t]
        scores = []
        for i in range(k):
            a_inv = np.linalg.inv(a_mats[i])
            th = a_inv @ b_vecs[i]
            scores.append(float(x @ th) + alpha * np.sqrt(x @ a_inv @ x))
        a = int(np.argmax(scores))
        r = float(x @ thetas[a]) + noise * float(rng.standard_normal())
        a_mats[a] += np.outer(x, x)
        b_vecs[a] += x * r
        picks.append(a)
    return _oracle_regret(thetas, ctx, picks)


def lin_ts(
    thetas: FloatArray,
    ctx: FloatArray,
    rng: np.random.Generator,
    noise: float = 0.25,
    v2: float = 1.0,
) -> float:
    """Linear Thompson sampling (disjoint Gaussian posterior)."""
    k, d = thetas.shape
    a_mats = np.stack([np.eye(d) for _ in range(k)])
    b_vecs = np.zeros((k, d))
    picks: list[int] = []
    for t in range(ctx.shape[0]):
        x = ctx[t]
        scores = []
        for i in range(k):
            cov = v2 * np.linalg.inv(a_mats[i])
            th = rng.multivariate_normal(cov @ b_vecs[i], cov)
            scores.append(float(x @ th))
        a = int(np.argmax(scores))
        r = float(x @ thetas[a]) + noise * float(rng.standard_normal())
        a_mats[a] += np.outer(x, x)
        b_vecs[a] += x * r
        picks.append(a)
    return _oracle_regret(thetas, ctx, picks)


def lin_eps_greedy(
    thetas: FloatArray,
    ctx: FloatArray,
    rng: np.random.Generator,
    eps: float = 0.1,
    noise: float = 0.25,
) -> float:
    """Greedy ridge with epsilon random exploration."""
    k, d = thetas.shape
    a_mats = np.stack([np.eye(d) for _ in range(k)])
    b_vecs = np.zeros((k, d))
    picks: list[int] = []
    for t in range(ctx.shape[0]):
        x = ctx[t]
        if rng.random() < eps:
            a = int(rng.integers(k))
        else:
            scores = [float(x @ np.linalg.solve(a_mats[i], b_vecs[i])) for i in range(k)]
            a = int(np.argmax(scores))
        r = float(x @ thetas[a]) + noise * float(rng.standard_normal())
        a_mats[a] += np.outer(x, x)
        b_vecs[a] += x * r
        picks.append(a)
    return _oracle_regret(thetas, ctx, picks)


def bench_contextual_bandits(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    k, d, steps, runs = 4, 6, 1200, 4
    lu = ts = eg = 0.0
    for _ in range(runs):
        thetas, ctx = _env(rng, k, d, steps)
        lu += linucb(thetas, ctx, rng)
        ts += lin_ts(thetas, ctx, rng)
        eg += lin_eps_greedy(thetas, ctx, rng)
    lu, ts, eg = lu / runs, ts / runs, eg / runs
    return {
        "synthetic_linucb_regret": lu,
        "synthetic_lin_ts_regret": ts,
        "synthetic_eps_greedy_regret": eg,
        "synthetic_beats_greedy": 1.0 if min(lu, ts) < eg else 0.0,
    }
