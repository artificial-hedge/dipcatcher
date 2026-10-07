"""MAML-style meta-learning for fast portfolio adaptation (Exec-Summary (SYNTHETIC)
meta/continual item). A base allocator is meta-trained across synthetic
regime tasks so that a few gradient steps on a NEW regime recover a
near-optimal allocation — model-agnostic meta-learning (first-order).

Synthetic bench: task distribution = trending / mean-reverting / volatile
markets; meta-init adapts in <=5 steps vs from-scratch and vs a pooled model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

_N_ASSETS = 6


def make_task(kind: str, T: int, rng: np.random.Generator) -> FloatArray:
    """Return (T, n) return panel for a regime kind. Tasks share structure:
    each regime's optimal tilt lives in a common 2-factor subspace so a good
    init is worth days of data."""
    n = _N_ASSETS
    shared = np.array([0.03, 0.025, -0.025, -0.02, 0.02, -0.015])
    c = {"trend": 1.0, "revert": 0.6, "volatile": 1.4}[kind]
    drift = c * shared + rng.uniform(-0.5, 0.5, n) * 0.004
    if kind == "revert":
        px = np.zeros((T, n))
        for t in range(1, T):
            px[t] = -0.4 * px[t - 1] + drift + 0.05 * rng.standard_normal(n)
        return px
    if kind == "volatile":
        return drift + 0.05 * rng.standard_t(4, (T, n))
    return drift + 0.05 * rng.standard_normal((T, n))


def task_loss(w: FloatArray, rets: FloatArray) -> tuple[float, FloatArray]:
    """Negative risk-adjusted reward: -mean(w·r) + lam * ||w||^2."""
    port = rets @ w
    loss = -float(port.mean()) + 0.5 * float(np.sum(w * w))
    grad = -rets.mean(0) + w
    return loss, np.asarray(grad)


@dataclass
class Maml:
    n: int = _N_ASSETS
    inner_lr: float = 0.01
    meta_lr: float = 0.005

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.n)

    def adapt(
        self, rets_support: FloatArray, steps: int, theta: FloatArray | None = None
    ) -> FloatArray:
        w = self.theta.copy() if theta is None else np.asarray(theta, float).copy()
        for _ in range(steps):
            _, g = task_loss(w, rets_support)
            w -= self.inner_lr * g
        return w

    def meta_train(self, tasks: list[tuple[FloatArray, FloatArray]], epochs: int = 60) -> None:
        """First-order MAML: meta-grad = query loss grad at adapted weights."""
        for _ in range(epochs):
            meta_g = np.zeros(self.n)
            for sup, qry in tasks:
                w_ad = self.adapt(sup, 5)
                _, g = task_loss(w_ad, qry)
                meta_g += g
            self.theta -= self.meta_lr * meta_g / len(tasks)


def alloc_score(w: FloatArray, rets: FloatArray) -> float:
    port = rets @ w
    return float(port.mean() / (port.std() + 1e-9))


def bench_maml_portfolio(seed: int = 11) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["trend", "revert", "volatile"]
    tasks = []
    for k in kinds:
        for _ in range(6):
            panel = make_task(k, 80, rng)
            tasks.append((panel[:8], panel[8:]))
    m = Maml()
    m.meta_train(tasks)
    # held-out task of each kind
    adapted_scores, scratch_scores, pooled_scores = [], [], []
    pooled = np.zeros(_N_ASSETS)
    for sup, _qry in tasks:
        _, g = task_loss(pooled, sup)
        pooled -= 0.02 * g
    for k in kinds:
        panel = make_task(k, 80, rng)
        sup, qry = panel[:8], panel[8:]
        adapted_scores.append(alloc_score(m.adapt(sup, 5), qry))
        scratch = np.zeros(_N_ASSETS)
        for _ in range(5):
            _, g = task_loss(scratch, sup)
            scratch -= m.inner_lr * g
        scratch_scores.append(alloc_score(scratch, qry))
        pooled_scores.append(alloc_score(pooled.copy(), qry))
    return {
        "synthetic_maml_adapted_score": float(np.mean(adapted_scores)),
        "synthetic_maml_scratch_score": float(np.mean(scratch_scores)),
        "synthetic_maml_pooled_score": float(np.mean(pooled_scores)),
        "synthetic_maml_margin_vs_scratch": float(
            np.mean(adapted_scores) - np.mean(scratch_scores)
        ),
        "synthetic_maml_tasks": float(len(tasks)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_maml_portfolio(), indent=1))
