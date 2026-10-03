"""Speculative decoding (Leviathan et al. 2023).

A cheap draft model proposes γ tokens; the target model verifies them
in one pass with rejection sampling — accepted tokens match the target
distribution exactly. Reports acceptance rate, tokens-per-forward gain,
and the correctness check (sampled distribution == target).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bench_speculative_decoding(
    seed: int = 151,
    n_seq: int = 200,
    vocab: int = 12,
    ctx: int = 8,
    gamma: int = 4,
    draft_err: float = 0.35,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((ctx, vocab))
    ctx_seq = rng.integers(0, vocab, (n_seq, ctx))
    logits = ctx_seq @ w
    p_target = np.exp(logits - logits.max(-1, keepdims=True))
    p_target /= p_target.sum(-1, keepdims=True)
    p_draft = np.exp(
        logits
        + draft_err * rng.standard_normal(logits.shape)
        - (logits + draft_err * rng.standard_normal(logits.shape)).max(-1, keepdims=True)
    )
    p_draft /= p_draft.sum(-1, keepdims=True)

    accepted = 0
    total_proposed = 0
    target_calls = 0
    sampled = np.zeros((n_seq, vocab))
    for i in range(n_seq):
        t = p_target[i]
        d = p_draft[i]
        target_calls += 1
        for _g in range(gamma):
            tok = rng.choice(vocab, p=d)
            total_proposed += 1
            if rng.uniform() < min(1.0, t[tok] / max(d[tok], 1e-12)):
                accepted += 1
                sampled[i, tok] += 1
            else:
                tok_r = rng.choice(vocab, p=np.clip(t - d, 0, None) / np.clip(t - d, 0, None).sum())
                sampled[i, tok_r] += 1
                break
    acc_rate = accepted / max(total_proposed, 1)
    tok_per_fwd = 1 + accepted / target_calls / max(1, gamma / gamma)
    tv_err = float(
        np.abs(sampled / np.maximum(sampled.sum(-1, keepdims=True), 1) - p_target).mean()
    )
    speedup_naive = (accepted + target_calls) / target_calls
    return {
        "synthetic_spec_accept_rate": float(acc_rate),
        "synthetic_spec_speedup": float(speedup_naive),
        "synthetic_spec_tv_err": tv_err,
        "synthetic_spec_tokens_per_fwd": float(tok_per_fwd),
    }
