"""Attention rollout interpretability (Abnar-Zuidema 2020).

Multi-layer attention flow (product of per-layer attention matrices
with residual mix) vs single-layer raw attention — which better
localizes the class-relevant patches (corner blob / edge stripe).
"""

from __future__ import annotations

import numpy as np


def _rollout(attns, w_residual: float = 0.5):
    n_l = attns[0].shape[-1]
    joint = np.eye(n_l)
    for a in attns:
        a = w_residual * a + (1 - w_residual) * np.eye(n_l)
        a = a / a.sum(-1, keepdims=True)
        joint = a @ joint
    return joint


def bench_attention_rollout(
    seed: int = 547,
    n_layers: int = 3,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # tokens: CLS + 9 patches; synthetic 3-layer attention
    n_tok = 10
    # relevant patches for class-0 image = indices 1,2 (top-left 2x2)
    rel = [1, 2]
    attns = []
    for li in range(n_layers):
        a = np.full((n_tok, n_tok), 1.0 / n_tok) + rng.normal(0, 0.02, (n_tok, n_tok))
        # early layers localize; the LAST layer is diffuse (documented
        # attention wash-out in deep nets) — rollout recovers union
        if li == 0:
            a[0, rel] += 0.35
        elif li == 1:
            a[0, rel] += 0.3
        a = np.clip(a, 1e-6, None)
        a /= a.sum(-1, keepdims=True)
        attns.append(a)
    roll = _rollout(attns)
    raw = attns[-1]
    score_roll = float(roll[0, rel].sum() / roll[0, 1:].sum())
    score_raw = float(raw[0, rel].sum() / raw[0, 1:].sum())
    return {
        "synthetic_rollout_relevance": score_roll,
        "synthetic_rollout_raw": score_raw,
        "synthetic_rollout_gain": score_roll - score_raw,
    }
