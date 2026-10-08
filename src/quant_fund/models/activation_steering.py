"""Activation steering / representation engineering (Turner et al.) (SYNTHETIC).

Compute a steering vector as the mean-difference direction between
activations with/without feature k; adding λ·v flips downstream
feature expression. Measured: flip rate of a linear probe under
steering vs the no-steer control.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._interp_synth import feature_directions, synth_activations


def bench_activation_steering(
    seed: int = 269,
    n: int = 2000,
    lam: float = 2.0,
    target: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dirs = feature_directions(rng)
    a, s = synth_activations(n, dirs, rng)
    probe = LogisticRegression(max_iter=400).fit(a[: n // 2], s[: n // 2, target] > 0)
    acc0 = float(probe.score(a[n // 2 :], s[n // 2 :, target] > 0))
    # steering vector: mean activation diff between s_k>0 and s_k=0
    v = a[s[:, target] > 0].mean(0) - a[s[:, target] == 0].mean(0)
    v = v / np.linalg.norm(v)
    a_steer = a[n // 2 :] - lam * v[None, :]  # subtract feature
    flip = float((probe.predict(a_steer) != (s[n // 2 :, target] > 0)).mean())
    wrong_dir = rng.normal(0, 1, a.shape[1])
    wrong_dir /= np.linalg.norm(wrong_dir)
    a_bad = a[n // 2 :] - lam * wrong_dir[None, :]
    flip_bad = float((probe.predict(a_bad) != (s[n // 2 :, target] > 0)).mean())
    return {
        "synthetic_steer_flip_rate": flip,
        "synthetic_steer_probe_acc": acc0,
        "synthetic_steer_wrongdir_flip": flip_bad,
        "synthetic_steer_gain": flip - flip_bad,
    }
