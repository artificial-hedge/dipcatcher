"""Counterparty credit-risk graph network (Exec-Summary graph item). Nodes are
counterparties with balance-sheet features; edges are exposure links. A message-
passing network learns default risk propagation — a node's risk depends on its
neighbors' health (Eisenberg-Noe-style contagion, learned).

Synthetic bench: interbank network where defaults cascade along edges;
the GNN's predicted distress beats a node-feature-only model OOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def synth_system(
    n: int, T: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return node features (n,4) per period, exposure graph (n,n), and
    realized distress labels (T,n): distress arises from own leverage plus
    neighbors' distress (cascade)."""
    lev = rng.uniform(0.1, 0.9, n)  # leverage
    liq = rng.uniform(0.1, 0.9, n)  # liquidity buffer
    size = rng.uniform(0.5, 1.5, n)
    # sparse directed exposures: i exposed to j
    adj = (rng.random((n, n)) < 0.15).astype(float)
    np.fill_diagonal(adj, 0)
    adj *= rng.uniform(0.2, 1.0, (n, n))
    feats = np.column_stack([lev, liq, size, np.ones(n)])
    dist = np.zeros((T, n))
    dprev = rng.random(n) < lev * 0.2
    for t in range(T):
        shock = rng.random(n) < np.clip(lev - 0.3 * liq, 0, 1) * 0.04
        spill = adj @ dprev > 0.25  # distressed neighbors transmit
        dprev = (shock | spill) & (rng.random(n) > liq * 0.5)
        dist[t] = dprev
    return feats, adj, dist


@dataclass
class MessageNet:
    """One message-passing round + logistic readout, closed-form-ish fit.

    node_score = W · [feat | A_norm @ feat | A_norm @ distress_lag]
    Trained by logistic IRLS-lite (gradient steps on all rows).
    """

    seed: int = 0

    def __post_init__(self) -> None:
        self.w = np.zeros(9)

    def _features(self, feats: FloatArray, adj_n: FloatArray, dlag: FloatArray) -> FloatArray:
        return np.column_stack([feats, adj_n @ feats, adj_n @ dlag])

    def fit(self, x: FloatArray, y: FloatArray, lr: float = 0.5, iters: int = 400) -> None:
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ self.w, -20, 20)))
            g = x.T @ (p - y) / len(y)
            self.w -= lr * g

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x @ self.w, -20, 20))))


def auc_score(y: FloatArray, s: FloatArray) -> float:
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    npos = int(pos.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bench_counterparty_gnn(seed: int = 9) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, T = 30, 240
    feats, adj, dist = synth_system(n, T, rng)
    d = adj.sum(1, keepdims=True)
    adj_n = adj / np.maximum(d, 1e-9)
    net = MessageNet()
    tr = 160
    dlag = np.vstack([np.zeros(n), dist[:-1]])
    x_tr = np.vstack([net._features(feats, adj_n, dlag[t]) for t in range(tr)])
    y_tr = dist[:tr].ravel()
    x_te = np.vstack([net._features(feats, adj_n, dlag[t]) for t in range(tr, T)])
    y_te = dist[tr:].ravel()
    net.fit(x_tr, y_tr)
    s_gnn = net.predict(x_te)
    # node-only baseline: features without neighbor terms
    b = np.zeros(4)
    xb_tr = np.tile(feats, (tr, 1))
    xb_te = np.tile(feats, (T - tr, 1))
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb_tr @ b, -20, 20)))
        b -= 0.5 * xb_tr.T @ (p - y_tr) / len(y_tr)
    s_bl = np.asarray(1.0 / (1.0 + np.exp(-np.clip(xb_te @ b, -20, 20))))
    return {
        "synthetic_cgnn_auc": auc_score(y_te, s_gnn),
        "synthetic_cgnn_baseline_auc": auc_score(y_te, s_bl),
        "synthetic_cgnn_auc_margin": auc_score(y_te, s_gnn) - auc_score(y_te, s_bl),
        "synthetic_cgnn_edge_density": float(np.mean(adj > 0)),
        "synthetic_cgnn_distress_rate": float(dist[tr:].mean()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_counterparty_gnn(), indent=1))
