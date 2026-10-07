"""Unit tests for quant_fund.models._interp_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._interp_synth import (
    D_ACT,
    N_FEAT,
    feature_directions,
    synth_activations,
)


def test_feature_directions_unit_norm() -> None:
    f = feature_directions(np.random.default_rng(0))
    assert f.shape == (N_FEAT, D_ACT)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), np.ones(N_FEAT), atol=1e-12)


def test_feature_directions_deterministic() -> None:
    np.testing.assert_array_equal(
        feature_directions(np.random.default_rng(3)),
        feature_directions(np.random.default_rng(3)),
    )


def test_activations_reconstruct_from_s() -> None:
    rng = np.random.default_rng(0)
    dirs = feature_directions(np.random.default_rng(1))
    a, s = synth_activations(200, dirs, rng)
    assert a.shape == (200, D_ACT) and s.shape == (200, N_FEAT)
    # a ≈ s @ dirs + small noise
    np.testing.assert_allclose(a, s @ dirs, atol=0.15)


def test_activations_sparsity_rate() -> None:
    rng = np.random.default_rng(0)
    dirs = feature_directions(np.random.default_rng(1))
    _a, s = synth_activations(2000, dirs, rng, sparsity=0.7)
    assert (s == 0).mean() == pytest.approx(0.7, abs=0.03)


def test_activations_reject_out_of_range_sparsity() -> None:
    # sparsity>1 zeroes every coefficient -> activations are pure noise
    # presented as feature-driven data. Must raise.
    rng = np.random.default_rng(0)
    dirs = feature_directions(np.random.default_rng(1))
    with pytest.raises(ValueError, match="sparsity"):
        synth_activations(8, dirs, rng, sparsity=1.5)
    with pytest.raises(ValueError, match="sparsity"):
        synth_activations(8, dirs, rng, sparsity=-0.1)
