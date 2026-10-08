"""Continual online learner with EWC-style consolidation (Exec-Summary (SYNTHETIC)
continual item). A linear signal model must learn a stream of regimes
WITHOUT forgetting earlier ones: elastic weight consolidation penalizes
drift on parameters important to past tasks (diagonal Fisher).

Synthetic bench: sequential regime stream (momentum -> reversal ->
volatility); EWC learner retains early-regime accuracy where plain SGD
catastrophically forgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FloatArray = np.ndarray


def regime_panel(kind: str, T: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Features x (T,4) -> next-period return y (T,). Each regime has a
    different true coefficient vector."""
    x = rng.standard_normal((T, 4))
    beta = {
        "momentum": np.array([0.9, 0.0, 0.0, 0.0]),
        "reversal": np.array([-0.8, 0.0, 0.3, 0.0]),
        "volatility": np.array([0.0, 0.7, 0.0, 0.4]),
    }[kind]
    y = x @ beta + 0.3 * rng.standard_normal(T)
    return x, y


@dataclass
class EwcLinear:
    n_feat: int = 4
    lam: float = 3.0
    lr: float = 0.05
    w: FloatArray = field(init=False)
    anchors: list[FloatArray] = field(default_factory=list)
    fishers: list[FloatArray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.w = np.zeros(self.n_feat)

    def train_task(self, x: FloatArray, y: FloatArray, epochs: int = 30) -> None:
        for _ in range(epochs):
            for i in range(len(y)):
                p = float(x[i] @ self.w)
                e = p - float(y[i])
                g = x[i] * e
                for wa, f in zip(self.anchors, self.fishers, strict=True):
                    g = g + self.lam * f * (self.w - wa)
                self.w -= self.lr * np.clip(g, -50, 50)
        self._consolidate(x, y)

    def _consolidate(self, x: FloatArray, y: FloatArray) -> None:
        f = np.mean(x**2, axis=0)  # diagonal Gauss-Newton Fisher proxy
        f = f / np.maximum(f.mean(), 1e-12)
        self.anchors.append(self.w.copy())
        self.fishers.append(np.asarray(f))

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(x @ self.w)


def mse(model: EwcLinear, x: FloatArray, y: FloatArray) -> float:
    return float(np.mean((model.predict(x) - y) ** 2))


def bench_continual_learning(seed: int = 13) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kinds = ["momentum", "reversal", "volatility"]
    tasks = {k: regime_panel(k, 60, rng) for k in kinds}
    ewc = EwcLinear()
    sgd = EwcLinear(lam=0.0)
    for k in kinds:
        x, y = tasks[k]
        ewc.train_task(x, y)
        sgd.train_task(x, y)
    # retention: MSE on EACH earlier task after full stream
    ewc_ret, sgd_ret = [], []
    for k in kinds:
        x, y = tasks[k]
        ewc_ret.append(mse(ewc, x, y))
        sgd_ret.append(mse(sgd, x, y))
    yv = np.concatenate([tasks[k][1] for k in kinds])
    base_var = float(np.var(yv))
    early = slice(0, len(kinds) - 1)
    return {
        "synthetic_ewc_early_retention_mse": float(np.mean(ewc_ret[early])),
        "synthetic_sgd_early_retention_mse": float(np.mean(sgd_ret[early])),
        "synthetic_ewc_forgetting_margin": float(np.mean(sgd_ret[early]) - np.mean(ewc_ret[early])),
        "synthetic_ewc_first_task_mse": ewc_ret[0],
        "synthetic_sgd_first_task_mse": sgd_ret[0],
        "synthetic_ewc_last_task_mse": ewc_ret[-1],
        "synthetic_sgd_last_task_mse": sgd_ret[-1],
        "synthetic_ewc_rel_var_last": float(np.mean(ewc_ret) / base_var),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_continual_learning(), indent=1))
