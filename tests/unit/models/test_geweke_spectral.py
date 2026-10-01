"""Tests for geweke_spectral (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.geweke_spectral import (
    bench_geweke_spectral,
    geweke_spectrum,
    synth_geweke,
)


def test_directional_asymmetry() -> None:
    pair, _ = synth_geweke(seed=1)
    sp = geweke_spectrum(pair, p=5)
    assert np.mean(sp["f_2to1"]) > 3 * np.mean(sp["f_1to2"])


def test_null_low() -> None:
    _, indep = synth_geweke(seed=2)
    sp = geweke_spectrum(indep, p=5)
    assert np.max(sp["f_2to1"]) < 0.15


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        geweke_spectrum(np.ones((50, 2)))
    with pytest.raises(ValueError):
        geweke_spectrum(np.ones(200))
    with pytest.raises(ValueError):
        geweke_spectrum(np.full((200, 2), np.nan))


def test_determinism() -> None:
    pair, _ = synth_geweke(seed=3)
    a = geweke_spectrum(pair, p=5)
    b = geweke_spectrum(pair, p=5)
    assert np.allclose(a["f_2to1"], b["f_2to1"])


def test_bench_schema_and_score() -> None:
    r = bench_geweke_spectral()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
