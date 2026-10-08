"""Tests for crossformer — two-stage attention over time and channel."""

from __future__ import annotations

import numpy as np

from quant_fund.models.crossformer import bench_crossformer, synth_patch_coupled


def test_patch_tokens_are_per_channel_segments() -> None:
    # token (c, p) must be channel c's contiguous timesteps — the old bare
    # reshape interleaved every channel into each "time" slot.
    from quant_fund.models.crossformer import _patch_tokens

    torch = __import__("torch")
    win, n_ch = 8, 2
    x = torch.arange(1 * win * n_ch, dtype=torch.float32).reshape(1, win, n_ch)
    tok = _patch_tokens(x, n_ch, 4)  # (1, 2, 4, 2)
    for c in range(n_ch):
        for p in range(4):
            seg = tok[0, c, p]
            assert torch.equal(seg, x[0, 2 * p : 2 * p + 2, c]), (c, p)


def test_synth_patch_coupled_shapes() -> None:
    rng = np.random.default_rng(0)
    x, y = synth_patch_coupled(16, 48, 4, rng)
    assert x.shape == (16, 48, 4)
    assert y.shape == (16,)


def test_bench_crossformer_smoke() -> None:
    # reduced iteration count keeps the smoke probe fast
    out = bench_crossformer(n=96, iters=20)
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
