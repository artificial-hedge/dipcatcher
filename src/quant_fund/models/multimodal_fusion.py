"""Multimodal data fusion forecaster (Exec-Summary Feature 6) (SYNTHETIC).
Modalities: price series (return stats), text (hashed-embed sentiment
probe), visual chart snapshot (gradient/momentum image stats). Gated
late-fusion learns per-modality weights; a fused logit predicts next-
day direction.

Synthetic bench: fused AUC vs best unimodal AUC on complementary
synthetic signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def price_feats(px: FloatArray) -> FloatArray:
    r = np.diff(px) / px[:-1]
    k = min(20, len(r))
    tail = r[-k:]
    return np.array([tail.mean(), tail.std(), np.sign(tail[-3:].sum()), np.tanh(50 * tail.sum())])


def chart_feats(px: FloatArray) -> FloatArray:
    """Rasterize the tail of px into a 16x16 gradient image, then stats."""
    img = np.zeros((16, 16))
    tail = px[-64:]
    xs = np.linspace(0, 15, len(tail)).astype(int)
    ys = ((tail - tail.min()) / (np.ptp(tail) + 1e-9) * 14).astype(int)
    img[15 - ys, xs] = 1.0
    gy, gx = np.gradient(img)
    return np.array([gx.sum(), gy.sum(), img[:, 8:].mean() - img[:, :8].mean(), img.mean()])


def text_feats(headline: str, dim: int = 16) -> FloatArray:
    v = np.zeros(dim)
    for tok in headline.lower().split():
        v[hash(tok) % dim] += 1.0
    return v


def sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


@dataclass
class GatedFusion:
    """Learns gate weights g_m and a fused logit over modality scores."""

    lr: float = 0.2

    def fit(
        self,
        scores: FloatArray,  # (n, m) per-modality scores
        y: FloatArray,
        iters: int = 400,
    ) -> None:
        n, m = scores.shape
        self.gate = np.ones(m) / m
        self.w = np.zeros(m)
        self.b = 0.0
        for _ in range(iters):
            g = np.exp(self.gate) / np.exp(self.gate).sum()
            z = (scores * g) @ self.w + self.b
            p = sigmoid(z)
            err = (y - p) / n
            self.w += self.lr * (scores.T @ err)
            self.b += self.lr * err.sum()
            # gate gradient: push weight to the modality with aligned sign
            corr = (scores.T * (y - 0.5)).mean(axis=1)
            self.gate += 0.1 * self.lr * corr

    def predict(self, scores: FloatArray) -> FloatArray:
        g = np.exp(self.gate) / np.exp(self.gate).sum()
        return sigmoid((scores * g) @ self.w + self.b)


def auc(y: FloatArray, p: FloatArray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(y))
    ranks[order] = np.arange(1, len(y) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def synth_scenario(n: int, rng: np.random.Generator) -> dict[str, FloatArray]:
    """Latent state drives price drift, news tone, and chart slope."""
    latent = np.tanh(rng.standard_normal(n))
    y = (latent + 0.6 * rng.standard_normal(n) > 0).astype(float)
    px_scores = np.tanh(0.9 * latent + 0.5 * rng.standard_normal(n))
    tx_scores = np.tanh(1.1 * latent + 0.4 * rng.standard_normal(n))
    ch_scores = np.tanh(0.7 * latent + 0.8 * rng.standard_normal(n))
    return {
        "scores": np.column_stack([px_scores, tx_scores, ch_scores]),
        "y": y,
    }


def bench_multimodal_fusion(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = synth_scenario(2000, rng)
    s, y = d["scores"], d["y"]
    tr, te = slice(0, 1200), slice(1200, 2000)
    uni = [auc(y[te], s[te, m]) for m in range(3)]
    fus = GatedFusion()
    fus.fit(s[tr], y[tr])
    p_f = fus.predict(s[te])
    fused_auc = auc(y[te], p_f)
    # modality ablation: drop text -> degradation
    fus2 = GatedFusion()
    fus2.fit(s[tr][:, [0, 2]], y[tr])
    abl = auc(y[te], fus2.predict(s[te][:, [0, 2]]))
    return {
        "synthetic_fusion_auc": fused_auc,
        "synthetic_fusion_best_uni_auc": float(max(uni)),
        "synthetic_fusion_gain_over_uni": fused_auc - max(uni),
        "synthetic_fusion_drop_text_auc": abl,
        "synthetic_fusion_text_increment": fused_auc - abl,
        "synthetic_fusion_gate_entropy": float(
            -np.sum(
                np.exp(fus.gate)
                / np.exp(fus.gate).sum()
                * np.log(np.exp(fus.gate) / np.exp(fus.gate).sum() + 1e-9)
            )
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_multimodal_fusion(), indent=1))
