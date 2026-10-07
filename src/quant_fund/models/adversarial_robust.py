"""Adversarial robustness for signal models (Exec-Summary robustness item) (SYNTHETIC).
FGSM perturbations on the input window degrade a plain return-sign
classifier; adversarial training (min-max on perturbed examples) restores
accuracy on a nonlinear MLP where robustness gains are real.

Synthetic bench: clean vs attacked accuracy for a standard and an
adversarially-trained MLP; eps in feature-sigma units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def synth_signal(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    x = rng.standard_normal((n, 8))
    beta = np.array([0.9, -0.7, 0.5, 0.3, -0.2, 0.1, 0.4, -0.3])
    z = x @ beta + 0.4 * x[:, 0] * x[:, 1] - 0.3 * x[:, 2] ** 2 + 0.5 * rng.standard_normal(n)
    return x, (z > 0).astype(float)


@dataclass
class Mlp:
    n_feat: int = 8
    hid: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = 0.3 * rng.standard_normal((self.n_feat, self.hid))
        self.b1 = np.zeros(self.hid)
        self.w2 = 0.3 * rng.standard_normal(self.hid)
        self.b2 = 0.0

    def _f(self, x: FloatArray) -> FloatArray:
        return x @ self.w1 + self.b1

    def prob(self, x: FloatArray) -> FloatArray:
        h = np.tanh(self._f(x))
        z = h @ self.w2 + self.b2
        return np.asarray(1.0 / (1.0 + np.exp(-z)))

    def grad_x(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """dBCE/dx per sample: (p-y) * w2 * (1-tanh^2) @ w1^T."""
        p = self.prob(x)
        dz = (p - y)[:, None] * self.w2[None, :]
        dh = dz * (1 - np.tanh(self._f(x)) ** 2)
        return np.asarray(dh @ self.w1.T)

    def step(self, x: FloatArray, y: FloatArray, lr: float) -> None:
        h_in = self._f(x)
        h = np.tanh(h_in)
        p = self.prob(x)
        n = len(y)
        dz = p - y
        self.w2 -= lr * (h.T @ dz / n)
        self.b2 -= lr * float(dz.mean())
        dh = dz[:, None] * self.w2[None, :] * (1 - h**2)
        self.w1 -= lr * (x.T @ dh / n)
        self.b1 -= lr * dh.mean(0)

    def fit(
        self, x: FloatArray, y: FloatArray, lr: float = 0.1, epochs: int = 300, eps: float = 0.0
    ) -> None:
        for _ in range(epochs):
            xa = x if eps <= 0 else x + eps * np.sign(self.grad_x(x, y))
            self.step(xa, y, lr)

    def acc(self, x: FloatArray, y: FloatArray) -> float:
        return float(np.mean((self.prob(x) > 0.5) == (y > 0.5)))


def fgsm(model: Mlp, x: FloatArray, y: FloatArray, eps: float) -> FloatArray:
    return x + eps * np.sign(model.grad_x(x, y))


def bench_adversarial_robust(seed: int = 37) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y = synth_signal(1200, rng)
    x_tr, x_te = x[:900], x[900:]
    y_tr, y_te = y[:900], y[900:]
    plain = Mlp(seed=seed)
    plain.fit(x_tr, y_tr)
    robust = Mlp(seed=seed + 1)
    robust.fit(x_tr, y_tr, eps=0.2, epochs=400)
    eps = 0.2
    xa = fgsm(plain, x_te, y_te, eps)
    xr = fgsm(robust, x_te, y_te, eps)
    return {
        "synthetic_adv_plain_clean_acc": plain.acc(x_te, y_te),
        "synthetic_adv_plain_attacked_acc": plain.acc(xa, y_te),
        "synthetic_adv_robust_clean_acc": robust.acc(x_te, y_te),
        "synthetic_adv_robust_attacked_acc": robust.acc(xr, y_te),
        "synthetic_adv_robustness_margin": robust.acc(xr, y_te) - plain.acc(xa, y_te),
        "synthetic_adv_eps_sigma": eps,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_adversarial_robust(), indent=1))
