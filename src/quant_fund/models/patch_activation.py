"""Activation patching / causal tracing (Vig et al., Meng et al.).

Run A (feature present) → patched into run B (absent) at the site
of feature k: downstream probe flips to A's label. Patching the
feature direction vs a control direction quantifies the causal role.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._interp_synth import feature_directions, synth_activations


def bench_patch_activation(
    seed: int = 281,
    n: int = 2000,
    target: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dirs = feature_directions(rng)
    a, s = synth_activations(n, dirs, rng)
    probe = LogisticRegression(max_iter=400).fit(a[: n // 2], s[: n // 2, target] > 0)
    te = slice(n // 2, n)
    on = s[te, target] > 0
    v = dirs[target]
    # patch: force component along feature dir to "on" scale
    a_patch = a[te].copy()
    proj = a_patch @ v[:, None]
    a_patch = a_patch + (1.5 - proj) * v[None, :] * (~on)[:, None]
    patched_flips = float((probe.predict(a_patch)[~on] == 1).mean())
    # control: patch along random dir
    w = rng.normal(0, 1, a.shape[1])
    w = w / np.linalg.norm(w)
    a_ctrl = a[te] + (1.5 - (a[te] @ w[:, None])) * w[None, :] * (~on)[:, None]
    ctrl_flips = float((probe.predict(a_ctrl)[~on] == 1).mean())
    return {
        "synthetic_patch_flip": patched_flips,
        "synthetic_patch_control_flip": ctrl_flips,
        "synthetic_patch_gain": patched_flips - ctrl_flips,
    }
