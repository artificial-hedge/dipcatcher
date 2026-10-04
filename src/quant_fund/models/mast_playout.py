"""MAST — Move-Average Sampling Technique for playout policies.

Global per-action Q-values learned from playout rewards bias a Gibbs
softmax rollout policy toward historically good moves. Bench: MAST
playouts on a biased-bandit maze game beat uniform-random playouts on
expected reward.
"""

import numpy as np

_SEED = 20261231 + 881

# Game: corridor of length L. Moves: 0 = advance 1, 1 = advance 2 (60%
# success), 2 = self-destruct (instant loss). Reaching the end wins; the
# fewer steps, the higher the reward. Action 2 dominates credit: any
# playout taking it fails.
L_CORR = 12


def _rollout(
    rng: np.random.Generator, q: np.ndarray | None = None, temp: float = 1.0
) -> tuple[float, list[int]]:
    pos, steps, taken = 0, 0, []
    while pos < L_CORR and steps < 40:
        if q is None:
            m = int(rng.choice(3))
        else:
            qn = (q - q.min()) / (q.max() - q.min() + 1e-12)
            logits = qn / max(temp, 1e-6)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            m = int(rng.choice(3, p=p))
        taken.append(m)
        if m == 2:
            return 0.0, taken
        if m == 0:
            pos += 1
        else:
            pos += 2 if rng.random() < 0.6 else 0
        steps += 1
    return (1.0 - 0.02 * steps) if pos >= L_CORR else 0.0, taken


def mast_train(iters: int = 400, seed: int = _SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_a = np.zeros(3)
    q_a = np.zeros(3)
    for _ in range(iters):
        r, taken = _rollout(rng, q_a, temp=0.5)
        for m in set(taken):
            n_a[m] += 1
            q_a[m] += (r - q_a[m]) / n_a[m]
    return q_a


def bench_mast_playout(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: MAST-softmax playouts beat uniform playouts on mean reward."""
    rng = np.random.default_rng(seed + 1)
    q = mast_train(seed=seed)
    r_mast = np.mean([_rollout(rng, q)[0] for _ in range(300)])
    r_rand = np.mean([_rollout(rng)[0] for _ in range(300)])
    return {"synthetic_mast_playout": 1.0 if r_mast > r_rand + 0.05 else 0.0}
