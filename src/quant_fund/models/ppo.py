"""Proximal Policy Optimization (Schulman et al., 2017) — clipped (SYNTHETIC)
surrogate objective L^CLIP = min(r·Â, clip(r)·Â) and a minibatch
gradient update on a softmax bandit policy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ppo_loss(ratios: FloatArray, adv: FloatArray, clip: float = 0.2) -> FloatArray:
    """Per-sample clipped PPO loss (to minimize: −L^CLIP)."""
    clipped = np.clip(ratios, 1.0 - clip, 1.0 + clip)
    out: FloatArray = -np.minimum(ratios * adv, clipped * adv)
    return out


def ppo_update_bandit(
    logits: FloatArray,
    actions: FloatArray,
    adv: FloatArray,
    lr: float = 0.05,
    clip: float = 0.2,
    epochs: int = 4,
) -> FloatArray:
    """PPO epochs on samples drawn from the OLD policy.

    actions: sampled action indices under π_old (logits_before).
    adv: per-sample advantages. Computes the clipped-surrogate
    gradient ∂/∂θ Σ −min(rÂ, clip rÂ) at the sampled actions via the
    score-function form (active branch only) and applies lr steps.
    """
    logits_old = logits.copy()
    pi_old = softmax_bandit(logits_old)
    theta = logits.copy()
    for _ in range(epochs):
        pi = softmax_bandit(theta)
        for i, a in enumerate(actions.astype(int)):
            r = pi[a] / pi_old[a]
            surr_new = r * adv[i]
            surr_clip = np.clip(r, 1 - clip, 1 + clip) * adv[i]
            if surr_new <= surr_clip:  # unclipped branch active
                grad = -adv[i] / pi_old[a] * (np.eye(len(theta))[a] - pi)
                theta = theta - lr * grad
    out: FloatArray = np.asarray(theta)
    return out


def softmax_bandit(logits: FloatArray) -> FloatArray:
    z = logits - logits.max()
    e = np.exp(z)
    out: FloatArray = np.asarray(e / e.sum())
    return out


def bench_ppo(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: loss plateau outside clip region; bandit updates
    raise the best arm's probability without overshooting."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # clip behavior: positive adv — loss flat beyond 1+ε
    ratios = np.linspace(0.5, 2.0, 30)
    adv_pos = np.ones(30) * 0.8
    loss = ppo_loss(ratios, adv_pos, 0.2)
    out["synthetic_ppo_clip_plateau"] = float(
        np.abs(loss[ratios > 1.2] - loss[ratios > 1.2][0]).max()
    )
    # negative adv — loss flat below 1−ε
    loss_n = ppo_loss(ratios, -adv_pos, 0.2)
    out["synthetic_ppo_clip_plateau_neg"] = float(
        np.abs(loss_n[ratios < 0.8] - loss_n[ratios < 0.8][0]).max()
    )
    # bandit: true rewards [0.1, 0.9, 0.2]; collect samples, update
    rewards = np.array([0.1, 0.9, 0.2])
    logits = np.zeros(3)
    pi0 = softmax_bandit(logits)
    for _ in range(30):
        acts = rng.choice(3, 200, p=softmax_bandit(logits))
        advs = rewards[acts] - rewards @ softmax_bandit(logits)
        logits = ppo_update_bandit(logits, acts.astype(np.float64), advs, lr=0.02)
    pi1 = softmax_bandit(logits)
    out["synthetic_ppo_prob_gain"] = float(pi1[1] - pi0[1])
    out["synthetic_ppo_best_arm"] = float(np.argmax(pi1) == 1)
    return out


if __name__ == "__main__":
    print(bench_ppo())
