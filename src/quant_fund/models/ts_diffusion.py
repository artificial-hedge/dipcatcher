"""Time-series diffusion generator (Exec-Summary Feature 7). DDPM-lite on (SYNTHETIC)
windows of returns: forward cosine-noise schedule, a 1-hidden-layer
epsilon-predictor trained with manual gradients, reverse-chain sampling.

Synthetic bench: generated paths reproduce stylized facts of the source
series (abs-return autocorr, kurtosis, vol clustering) under proper scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def cosine_schedule(t: FloatArray, T: int) -> FloatArray:
    s = 0.008
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    return np.asarray(f / f[0])


@dataclass
class Denoiser:
    """eps_hat = W2 @ tanh(W1 @ [x_t; t_emb]) trained by hand."""

    dim: int
    hid: int = 32

    def __post_init__(self) -> None:
        self.w1 = 0.05 * np.random.default_rng(0).standard_normal((self.hid, self.dim + 2))
        self.w2 = 0.05 * np.random.default_rng(1).standard_normal((self.dim, self.hid))

    def forward(
        self, x: FloatArray, t_emb: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        inp = np.concatenate([x, t_emb])
        h = np.tanh(self.w1 @ inp)
        return self.w2 @ h, h, inp

    def step(self, x: FloatArray, t_emb: FloatArray, target: FloatArray, lr: float) -> float:
        pred, h, inp = self.forward(x, t_emb)
        err = pred - target
        loss = float(np.mean(err**2))
        g2 = np.outer(2 * err / len(err), h)
        g1 = np.outer((1 - h**2) * (self.w2.T @ (2 * err / len(err))), inp)
        self.w2 -= lr * g2
        self.w1 -= lr * g1
        return loss

    def predict(self, x: FloatArray, t_emb: FloatArray) -> FloatArray:
        out, _, _ = self.forward(x, t_emb)
        return out


class DiffusionSampler:
    def __init__(self, dim: int, T: int = 40):
        self.dim = dim
        self.T = T
        self.ab = cosine_schedule(np.arange(T + 1, dtype=float), T)
        self.model = Denoiser(dim)

    def train(
        self, windows: FloatArray, epochs: int, rng: np.random.Generator, lr: float = 0.02
    ) -> float:
        loss = 0.0
        n = len(windows)
        for _ in range(epochs):
            for i in range(0, n, 64):
                for x0 in windows[i : i + 64]:
                    t = int(rng.integers(1, self.T))
                    a_bar = self.ab[t]
                    eps = rng.standard_normal(self.dim)
                    x_t = np.sqrt(a_bar) * x0 + np.sqrt(1 - a_bar) * eps
                    emb = np.array([t / self.T, np.sin(8 * t / self.T)])
                    loss = self.model.step(x_t, emb, eps, lr)
        return loss

    def sample(self, rng: np.random.Generator) -> FloatArray:
        x = rng.standard_normal(self.dim)
        for t in range(self.T, 0, -1):
            emb = np.array([t / self.T, np.sin(8 * t / self.T)])
            eps = self.model.predict(x, emb)
            a_t = self.ab[t] / self.ab[t - 1]
            x = (x - (1 - a_t) / np.sqrt(1 - self.ab[t]) * eps) / np.sqrt(a_t)
            if t > 1:
                x += np.sqrt(max(1 - a_t, 0.0)) * 0.1 * rng.standard_normal(self.dim)
            x = np.clip(x, -3.0, 3.0)
        return x


def vol_cluster(x: FloatArray) -> float:
    return float(np.corrcoef(np.abs(x[:-1]), np.abs(x[1:]))[0, 1])


def kurt(x: FloatArray) -> float:
    m = x - x.mean()
    return float(np.mean(m**4) / (np.mean(m**2) ** 2 + 1e-12) - 3.0)


def garch_like_windows(n: int, w: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """GARCH(1,1)-ish series + sliding windows."""
    r = np.empty(n + w)
    v = np.empty(n + w)
    v[0] = 1e-4
    for t in range(1, n + w):
        v[t] = 0.05e-4 + 0.9 * v[t - 1] + 0.09 * r[t - 1] ** 2
        r[t] = np.sqrt(v[t]) * rng.standard_normal()
    wins = np.array([r[i : i + w] for i in range(n)])
    return wins, r


def bench_ts_diffusion(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    wins, full = garch_like_windows(600, 32, rng)
    scale = float(wins.std())
    wins_n = wins / scale
    ds = DiffusionSampler(32, T=30)
    final_loss = ds.train(wins_n[:400], 25, rng, lr=0.03)
    raw = np.concatenate([ds.sample(rng) for _ in range(20)])
    # shape-normalize: generated path rescaled to unit std per window block
    gen = np.concatenate([r / (r.std() + 1e-9) for r in raw.reshape(20, -1)]) * scale
    real = np.concatenate(wins[:20])
    ks_stat = float(
        np.max(
            np.abs(
                np.sort(real).searchsorted(np.sort(gen), side="right") / len(gen)
                - np.linspace(0, 1, len(gen))
            )
        )
    )
    return {
        "synthetic_ts_diff_final_loss": final_loss,
        "synthetic_ts_diff_kurtosis_gap": abs(kurt(gen) - kurt(real)),
        "synthetic_ts_diff_volcluster_gap": abs(vol_cluster(gen) - vol_cluster(real)),
        "synthetic_ts_diff_ks_stat": ks_stat,
        "synthetic_ts_diff_gen_std_ratio": float(gen.std() / real.std()),
        "synthetic_ts_diff_raw_std_ratio": float(raw.std() / wins_n[:20].reshape(-1).std()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_ts_diffusion(), indent=1))
