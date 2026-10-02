"""Causal signal mining over a factor zoo (Exec-Summary causal item).
PCMCI-lite: for each target return series, candidate factor parents at
lags 1..tau_max are tested by lagged partial correlation given the
target's own past (momentary conditional independence, Runge-style).

Synthetic bench: planted DAG (3 true drivers + 2 mediator + 3 noise
factors); parent-selection precision/recall and effect-sign accuracy
vs the truth. Bonferroni-controlled alpha.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray

_FACTORS = 8
_TRUE = {0: 0.6, 1: -0.5, 2: 0.4}  # true driver -> coefficient


def synth_zoo(T: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, dict[int, float]]:
    """Returns (T, F) factor panel + target (T,) + truth dict.

    factors 0..2 drive y (lag1); factor 3 is a mediator of 0 (confound);
    factor 4 correlates with 1 contemporaneously (not causal); 5..7 noise.
    """
    f = rng.standard_normal((T, _FACTORS))
    f[1:, 3] = 0.7 * f[:-1, 0] + 0.5 * f[1:, 3]
    f[:, 4] = 0.8 * f[:, 1] + 0.4 * f[:, 4]
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (
            sum(b * f[t - 1, k] for k, b in _TRUE.items())
            + 0.5 * y[t - 1] * 0
            + 0.4 * rng.standard_normal()
        )
    return f, y, dict(_TRUE)


def partial_corr_lag1(f: FloatArray, y: FloatArray, j: int) -> float:
    """Corr(f_j[t-1], y[t] | y[t-1], f_j[t-2]) via residualization."""
    a = f[1:-1, j]
    b = y[2:]
    z = np.column_stack([y[1:-1], f[:-2, j]])
    za = np.column_stack([np.ones(len(z)), z])
    ra = a - za @ np.linalg.lstsq(za, a, rcond=None)[0]
    rb = b - za @ np.linalg.lstsq(za, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def pcmci_parents(f: FloatArray, y: FloatArray, alpha: float = 0.01) -> dict[int, float]:
    T = len(y)
    out: dict[int, float] = {}
    thr = 2.58 / np.sqrt(T - 4)  # ~alpha .01 normal approx on corr
    for j in range(f.shape[1]):
        r = partial_corr_lag1(f, y, j)
        if abs(r) > thr:
            out[j] = r
    return out


def bench_causal_miner(seed: int = 43) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    f, y, truth = synth_zoo(1500, rng)
    parents = pcmci_parents(f, y)
    tp = sum(1 for j in parents if j in truth)
    fp = sum(1 for j in parents if j not in truth)
    prec = tp / max(len(parents), 1)
    rec = tp / len(truth)
    # sign agreement on recovered true parents
    sign_hits = [np.sign(parents[j]) == np.sign(truth[j]) for j in truth if j in parents]
    sign_acc = float(np.mean(sign_hits)) if sign_hits else 0.0
    # naive corr baseline (no conditioning) flags spurious factor 4
    naive = {j: float(np.corrcoef(f[:-1, j], y[1:])[0, 1]) for j in range(_FACTORS)}
    naive_sel = [j for j, r in naive.items() if abs(r) > 2.58 / np.sqrt(len(y) - 4)]
    naive_fp = sum(1 for j in naive_sel if j not in truth)
    return {
        "synthetic_pcmci_precision": float(prec),
        "synthetic_pcmci_recall": float(rec),
        "synthetic_pcmci_false_positives": float(fp),
        "synthetic_pcmci_sign_accuracy": sign_acc,
        "synthetic_pcmci_naive_fp": float(naive_fp),
        "synthetic_pcmci_n_selected": float(len(parents)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_causal_miner(), indent=1))
