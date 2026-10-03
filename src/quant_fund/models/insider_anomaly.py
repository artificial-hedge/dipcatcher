"""Insider-flow anomaly detection (Exec-Summary surveillance item). A linear
autoencoder over order-flow features reconstructs "normal" flow; bursts of
informed trading produce high reconstruction error, detected with robust
z-scores + isolation-style random projections.

Synthetic bench: tape mixes background flow with rare informed bursts
(persistent aggressor side, concentrated sizes); detector AUC/precision
on the burst labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def synth_flow(n_windows: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """(n, 5) flow features + insider label. Bursts: high signed imbalance,
    large trades, low diversity, price drift."""
    x = np.zeros((n_windows, 5))
    y = rng.random(n_windows) < 0.1
    x[:, 0] = 0.1 * rng.standard_normal(n_windows)  # imbalance
    x[:, 1] = np.abs(rng.standard_normal(n_windows))  # trade size
    x[:, 2] = rng.random(n_windows)  # trade-count entropy proxy
    x[:, 3] = 0.05 * rng.standard_normal(n_windows)  # drift
    x[:, 4] = rng.random(n_windows)  # dark share
    x[y, 0] = np.sign(rng.standard_normal(int(y.sum()))) * rng.uniform(0.25, 0.6, int(y.sum()))
    x[y, 1] = rng.uniform(0.8, 2.2, int(y.sum()))
    x[y, 2] = rng.uniform(0.2, 0.6, int(y.sum()))
    x[y, 3] = np.sign(x[y, 0]) * rng.uniform(0.1, 0.3, int(y.sum()))
    x[y, 4] = rng.uniform(0.4, 0.9, int(y.sum()))
    return x, y.astype(float)


@dataclass
class LinearAe:
    """PCA-style autoencoder: project to k comps and back; reconstruction
    error = anomaly score."""

    k: int = 2

    def fit(self, x: FloatArray) -> None:
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-9
        z = (x - self.mu) / self.sd
        cov = z.T @ z / len(z)
        vals, vecs = np.linalg.eigh(cov)
        self.v = vecs[:, -self.k :]

    def score(self, x: FloatArray) -> FloatArray:
        z = (x - self.mu) / self.sd
        recon = (z @ self.v) @ self.v.T
        return np.asarray(np.mean((z - recon) ** 2, axis=1), dtype=float)


def robust_z(scores: FloatArray) -> FloatArray:
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) * 1.4826 + 1e-12
    return np.asarray((scores - med) / mad)


def auc_score(y: FloatArray, s: FloatArray) -> float:
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    npos = int(pos.sum())
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def bench_insider_anomaly(seed: int = 19) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x, y = synth_flow(600, rng)
    ae = LinearAe(k=2)
    ae.fit(x[y < 0.5])  # train on normal only
    s = robust_z(ae.score(x))
    auc = auc_score(y, s)
    thr = np.quantile(s, 0.9)
    flagged = s > thr
    prec = float(np.mean(y[flagged] > 0.5)) if flagged.any() else 0.0
    rec = float(np.mean(flagged[y > 0.5])) if (y > 0.5).any() else 0.0
    # volume-only baseline
    s_bl = robust_z(x[:, 1])
    return {
        "synthetic_insider_auc": auc,
        "synthetic_insider_precision_at10": prec,
        "synthetic_insider_recall_at10": rec,
        "synthetic_insider_volume_auc": auc_score(y, s_bl),
        "synthetic_insider_auc_margin": auc - auc_score(y, s_bl),
        "synthetic_insider_base_rate": float(y.mean()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_insider_anomaly(), indent=1))
