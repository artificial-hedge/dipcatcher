"""Trust-Region Policy Optimization (Schulman et al., 2015) — the
natural-gradient step constrained by a KL ball on the policy. For a
softmax policy the Fisher matrix is Cov_π(∇logπ) = diag(π) − ππᵀ and
the step solves F x = g conjugate-gradient-style, scaled to
½ xᵀFx ≤ δ_KL.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def softmax(logits: FloatArray) -> FloatArray:
    z = logits - logits.max()
    e = np.exp(z)
    out: FloatArray = np.asarray(e / e.sum())
    return out


def fisher_softmax(pi: FloatArray) -> FloatArray:
    """Fisher information for softmax logits: diag(π) − ππᵀ."""
    out: FloatArray = np.diag(pi) - np.outer(pi, pi)
    return out


def trpo_step(logits: FloatArray, adv: FloatArray, kl_max: float = 0.01) -> FloatArray:
    """One TRPO step on a context-free softmax policy.

    Surrogate gradient at current π: g = π ∘ (adv − πᵀadv).
    Natural gradient direction x ∝ F⁺g, scaled so ½ xᵀFx = kl_max.
    """
    pi = softmax(logits)
    g = pi * (adv - float(pi @ adv))
    F = fisher_softmax(pi)

    # solve F x = g in the mean-zero complement (F is rank n−1);
    # ridge jitter regularizes the trivial null direction
    x_dir = np.linalg.lstsq(F + np.eye(len(pi)) * 1e-10, g, rcond=None)[0]
    f_val = float(x_dir @ F @ x_dir)
    if f_val <= 0:
        return logits
    scale = float(np.sqrt(2.0 * kl_max / f_val))
    step = scale * x_dir
    new_logits = logits + step
    # hard KL check; backtrack if violated
    kl = float(pi @ (np.log(pi + 1e-12) - np.log(softmax(new_logits) + 1e-12)))
    bt = 1.0
    while kl > kl_max and bt > 1e-3:
        bt *= 0.5
        new_logits = logits + step * bt
        kl = float(pi @ (np.log(pi + 1e-12) - np.log(softmax(new_logits) + 1e-12)))
    out: FloatArray = np.asarray(new_logits)
    return out


def bench_trpo(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 3-armed bandit adv [0.1, 0.9, 0.0]; TRPO steps raise
    π(best arm) monotonically while each step's KL stays ≲ δ."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    adv = np.array([0.1, 0.9, 0.0])
    logits = rng.normal(0, 0.1, 3)
    kl_max = 0.01
    pi0 = softmax(logits)
    kl_violate = 0.0
    for _ in range(40):
        pi_before = softmax(logits)
        logits = trpo_step(logits, adv, kl_max)
        pi_after = softmax(logits)
        kl = float(pi_before @ (np.log(pi_before + 1e-12) - np.log(pi_after + 1e-12)))
        kl_violate = max(kl_violate, kl - kl_max)
    pi1 = softmax(logits)
    out["synthetic_trpo_prob_gain"] = float(pi1[1] - pi0[1])
    out["synthetic_trpo_kl_violation"] = float(kl_violate)
    out["synthetic_trpo_improves"] = float(pi1[1] > pi0[1])
    out["synthetic_trpo_best_arm"] = float(np.argmax(pi1) == 1)
    return out


if __name__ == "__main__":
    print(bench_trpo())
