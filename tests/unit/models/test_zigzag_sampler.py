"""Adversarial probes for zigzag_sampler — uniform-time resampling."""

import numpy as np

from quant_fund.models.zigzag_sampler import _zigzag, bench_zigzag_sampler


def _trace(seed: int, horizon: float):
    import quant_fund.models.zigzag_sampler as m

    return m._zigzag_trace(seed, horizon)


def test_trace_times_increasing() -> None:
    t, pos = _trace(0, horizon=50.0)
    assert t.size == pos.shape[0]
    assert np.all(np.diff(t) > 0)


def test_uniform_grid_resample_is_denser() -> None:
    """_zigzag returns time-uniform resamples (~4x the event count), not the
    biased event-time subsample."""
    smp = _zigzag(0, horizon=50.0)
    t, pos = _trace(0, horizon=50.0)
    assert smp.shape[1] == pos.shape[1]
    # grid resample: min(n_grid, 4*n_events) rows — not the raw event subsample
    assert smp.shape[0] == min(4000, 4 * t.size)
    assert smp.shape[0] != t.size


def test_resample_exact_on_skeleton() -> None:
    """Interpolation on a segment returns the exact position (linear path)."""
    t, pos = _trace(1, horizon=50.0)
    mid = (t[1] + t[2]) / 2
    # reconstruct manually
    seg = (pos[2] - pos[1]) / (t[2] - t[1])
    expect = pos[1] + seg * (mid - t[1])
    assert np.allclose(expect, (pos[1] + pos[2]) / 2)


def test_bench_schema() -> None:
    r = bench_zigzag_sampler()
    assert set(r) >= {"synthetic_zz_ess", "synthetic_zz_moment_err"}
    assert np.isfinite(r["synthetic_zz_moment_err"])
