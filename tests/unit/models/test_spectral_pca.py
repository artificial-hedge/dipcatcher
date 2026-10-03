"""Tests for spectral_pca (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.spectral_pca import (
    bench_spectral_pca,
    dynamic_pca,
    synth_dpca,
)


def test_factor_share_dominates() -> None:
    panel, noise = synth_dpca(seed=1)
    sp = dynamic_pca(panel, m=5)
    sn = dynamic_pca(noise, m=5)
    assert sp["eig1_share_mean"][0] > sn["eig1_share_mean"][0] + 0.2


def test_share_bounds() -> None:
    panel, _ = synth_dpca(seed=2)
    sp = dynamic_pca(panel, m=5)
    assert 0.0 < sp["eig1_share_mean"][0] <= 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        dynamic_pca(np.ones((3, 50)))
    with pytest.raises(ValueError):
        dynamic_pca(np.full((6, 300), np.nan))


def test_determinism() -> None:
    panel, _ = synth_dpca(seed=3)
    a = dynamic_pca(panel, m=5)
    b = dynamic_pca(panel, m=5)
    assert np.allclose(a["eig1_share"], b["eig1_share"])


def test_bench_schema_and_score() -> None:
    r = bench_spectral_pca()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
