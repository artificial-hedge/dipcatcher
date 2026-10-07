"""Logit lens / tuned lens (nostalgebraist, Belrose et al.) (SYNTHETIC).

Reads intermediate activations straight through the output head:
project each layer's residual stream to logits and measure answer
accuracy per layer — shows the model "decides" progressively. A tuned
linear map per layer improves early readout (tuned-lens uplift).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._interp_synth import feature_directions, synth_activations


def bench_logit_lens(
    seed: int = 277,
    n: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dirs = feature_directions(rng)
    a, s = synth_activations(n, dirs, rng, sparsity=0.5)
    # fake residual stream: layer0 = partial features (only s[:, :4]),
    # layer1 adds the rest — lens should show monotone readout gain
    a0 = s[:, :4] @ dirs[:4]
    a1 = a
    y = (s[:, 5] > 0).astype(int)  # feature only present in layer1
    cut = n // 2
    acc_early = LogisticRegression(max_iter=300).fit(a0[:cut], y[:cut]).score(a0[cut:], y[cut:])
    acc_late = LogisticRegression(max_iter=300).fit(a1[:cut], y[:cut]).score(a1[cut:], y[cut:])
    # tuned lens: affine map on early layer before probing
    W = np.linalg.lstsq(np.concatenate([a0[:cut], np.ones((cut, 1))], -1), a1[:cut], rcond=None)[0]
    a0_t = np.concatenate([a0, np.ones((n, 1))], -1) @ W
    acc_tuned = LogisticRegression(max_iter=300).fit(a0_t[:cut], y[:cut]).score(a0_t[cut:], y[cut:])
    return {
        "synthetic_lens_early_acc": float(acc_early),
        "synthetic_lens_late_acc": float(acc_late),
        "synthetic_lens_tuned_acc": float(acc_tuned),
        "synthetic_lens_tuned_gain": float(acc_tuned - acc_early),
        "synthetic_lens_depth_gain": float(acc_late - acc_early),
    }
