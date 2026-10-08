"""Adversarial probes for celp_encode."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.celp_encode import _lpc, _synth, bench_celp_encode, celp_frame


def _voiced_frame() -> tuple[np.ndarray, np.ndarray]:
    n = 160
    a = np.array([1.0, -0.8, 0.4, -0.15])
    exc_true = np.tile(np.r_[np.zeros(19), 3.0], n // 20)
    x = _synth(exc_true, a)
    return x, _lpc(x, 3)


def test_npitch_bounds_adaptive_search() -> None:
    # ``npitch`` bounds the adaptive-codebook lag window — the old code
    # accepted the parameter and ignored it entirely.
    x, a_hat = _voiced_frame()
    e40, _ = celp_frame(x, a_hat, npitch=40)
    e80, _ = celp_frame(x, a_hat, npitch=80)
    assert not np.array_equal(e40, e80)


def test_pitch_vector_periodic_extension() -> None:
    from quant_fund.models.celp_encode import _pitch_vector

    hist = np.array([0.0, 1.0, 2.0, 3.0])
    v = _pitch_vector(hist, 3, 8)
    # last 3 samples periodically extended to subframe length
    assert np.allclose(v, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])


def test_frame_length_fail_closed() -> None:
    x, a_hat = _voiced_frame()
    # a 157-sample frame cannot split into 4 equal subframes — the old
    # code silently dropped the tail instead of failing closed.
    with pytest.raises(ValueError):
        celp_frame(x[:157], a_hat)


def test_voiced_excitation_has_pitch_structure() -> None:
    x, a_hat = _voiced_frame()
    exc, _ = celp_frame(x, a_hat)
    ac = np.correlate(exc - exc.mean(), exc - exc.mean(), "full")[159:]
    ac = ac / max(ac[0], 1e-9)
    assert ac[20] > 0.3


def test_bench_passes() -> None:
    assert bench_celp_encode()["synthetic_celp_encode"] == 1.0
