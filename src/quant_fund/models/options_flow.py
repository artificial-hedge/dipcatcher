"""Options-flow AI detector (Exec-Summary Feature 4). Streaming tape (SYNTHETIC)
featurizer (greeks-style aggregates, size z-scores, block flags, moneyness,
side persistence) plus a logistic classifier separating informed flow from
hedging flow.

Synthetic bench: labeled tape where informed trades are large, directional,
OTM-biased — AUC, Brier, alert precision on flagged flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class OptionTrade:
    ts: int
    side: float  # +1 buy call / -1 buy put (aggressor direction)
    size: float
    moneyness: float  # strike/spot; >1 OTM call, <1 OTM put
    iv: float
    oi: float
    voi: float  # volume/open-interest ratio


def synth_tape(n: int, rng: np.random.Generator) -> tuple[list[OptionTrade], FloatArray]:
    """Informed (y=1): big size, |side| persistent, OTM bias, high voi, iv pop."""
    trades: list[OptionTrade] = []
    y = np.zeros(n)
    regime = 0.0
    for t in range(n):
        informed = rng.random() < 0.25
        y[t] = float(informed)
        regime = 0.9 * regime + rng.standard_normal()
        if informed:
            size = float(rng.uniform(300, 2000))
            side = float(np.sign(regime) or 1.0)
            moneyness = float(rng.uniform(1.05, 1.35))
            iv = float(rng.uniform(0.35, 0.9))
            voi = float(rng.uniform(0.8, 3.0))
        else:
            size = float(np.exp(rng.normal(4.5, 0.8)))
            side = float(rng.choice([-1.0, 1.0]))
            moneyness = float(rng.uniform(0.85, 1.15))
            iv = float(rng.uniform(0.15, 0.45))
            voi = float(rng.uniform(0.05, 0.6))
        trades.append(OptionTrade(t, side, size, moneyness, iv, 5000.0, voi))
    return trades, y


def featurize(trades: list[OptionTrade]) -> FloatArray:
    n = len(trades)
    sizes = np.array([tr.size for tr in trades])
    mu, sd = sizes.mean(), sizes.std() + 1e-9
    X = np.zeros((n, 6))
    persist = 0.0
    for i, tr in enumerate(trades):
        persist = 0.8 * persist + 0.2 * tr.side
        X[i] = [
            1.0,
            (tr.size - mu) / sd,
            tr.voi,
            tr.iv,
            abs(tr.moneyness - 1.0),
            abs(persist),
        ]
    return X


def fit_logistic(X: FloatArray, y: FloatArray, lr: float = 0.3, iters: int = 400) -> FloatArray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w += lr * (X.T @ (y - p)) / len(y)
    return w


def sigmoid(X: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30))))


def auc_score(y: FloatArray, p: FloatArray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(y))
    ranks[order] = np.arange(1, len(y) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bench_options_flow(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    trades, y = synth_tape(6000, rng)
    X = featurize(trades)
    n_tr = 4000
    w = fit_logistic(X[:n_tr], y[:n_tr])
    p = sigmoid(X[n_tr:], w)
    yh = y[n_tr:]
    alert = p > 0.6
    prec = float(yh[alert].mean()) if alert.any() else 0.0
    rec = float(alert[yh == 1].mean())
    brier = float(np.mean((p - yh) ** 2))
    return {
        "synthetic_options_flow_auc": auc_score(yh, p),
        "synthetic_options_flow_brier": brier,
        "synthetic_options_flow_alert_precision": prec,
        "synthetic_options_flow_alert_recall": rec,
        "synthetic_options_flow_alert_rate": float(alert.mean()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_options_flow(), indent=1))
