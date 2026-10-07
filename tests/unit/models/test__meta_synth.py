"""Adversarial probes for quant_fund.models._meta_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._meta_synth import sine_task


def test_sine_task_shapes_and_consistency() -> None:
    rng = np.random.default_rng(0)
    xs, ys, xq, yq = sine_task(rng, K=5, M=15)
    assert xs.shape == ys.shape == (5,)
    assert xq.shape == yq.shape == (15,)
    # support and query y share (A, phi): yq/sin(xq+phi) constant.
    # Recover A,phi from the support set via a small grid search and
    # verify the query targets are on the same sinusoid.
    best = None
    for A in np.linspace(0.1, 5.0, 200):
        for ph in np.linspace(0, np.pi, 200):
            err = np.abs(A * np.sin(xs + ph) - ys).max()
            if best is None or err < best[0]:
                best = (err, A, ph)
    assert best is not None and best[0] < 0.05
    _, A, ph = best
    assert np.abs(A * np.sin(xq + ph) - yq).max() < 0.05


def test_sine_task_deterministic() -> None:
    a = sine_task(np.random.default_rng(3))
    b = sine_task(np.random.default_rng(3))
    for u, v in zip(a, b, strict=True):
        assert np.array_equal(u, v)


def test_sine_task_hostile() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sine_task(rng, K=0)
    with pytest.raises(ValueError):
        sine_task(rng, M=0)
