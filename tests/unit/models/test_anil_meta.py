"""Tests for models/anil_meta.py — pooled baseline must pair per-task."""

from __future__ import annotations

import numpy as np


def test_pooled_baseline_pairs_per_task() -> None:
    """synthetic_anil_pooled_mse must be the mean of per-task linear
    baselines (not the last task's baseline alone)."""
    from quant_fund.models._meta_synth import sine_task
    from quant_fund.models.anil_meta import bench_anil_meta

    out = bench_anil_meta(seed=863, n_tasks=2, K=5)
    seed = 863
    pools = []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 4000 + i), K=5)
        w, _, _, _ = np.linalg.lstsq(np.stack([xs, np.ones(len(xs))], 1), ys, rcond=None)
        pools.append(float(((xq * w[0] + w[1] - yq) ** 2).mean()))
    expected = float(np.mean(pools))
    assert abs(out["synthetic_anil_pooled_mse"] - expected) < 1e-9
    # under the defect it equalled pools[-1] only
    assert abs(out["synthetic_anil_pooled_mse"] - pools[-1]) > 1e-6 or expected == pools[-1]
