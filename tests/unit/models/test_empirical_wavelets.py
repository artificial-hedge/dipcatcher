import numpy as np
import pytest

from quant_fund.models.empirical_wavelets import bench_empirical_wavelets, ewt


def _two_tone(rng, n=1024):
    t = np.arange(n)
    return (
        np.cos(2 * np.pi * 0.08 * t)
        + 0.8 * np.cos(2 * np.pi * 0.25 * t + 0.4)
        + 0.05 * rng.standard_normal(n)
    )


def test_ewt_shapes():
    rng = np.random.default_rng(0)
    out = ewt(_two_tone(rng), n_bands=3)
    assert out["modes"].shape[1] == 1024
    assert out["modes"].shape[0] >= 2
    assert np.isfinite(out["modes"]).all()


def test_ewt_mode_separation():
    rng = np.random.default_rng(1)
    out = ewt(_two_tone(rng), n_bands=3)
    modes = np.asarray(out["modes"])
    mags = np.abs(np.fft.rfft(modes, axis=1))
    freqs = np.fft.rfftfreq(1024)
    dom = freqs[np.argmax(mags, axis=1)]
    # both planted tones must be covered by some mode
    for p in (0.08, 0.25):
        assert min(abs(d - p) for d in dom) < 0.03


def test_ewt_reconstruction():
    rng = np.random.default_rng(2)
    x = _two_tone(rng)
    out = ewt(x, n_bands=3)
    recon = np.asarray(out["modes"]).sum(axis=0)
    err = np.linalg.norm(x - recon) / np.linalg.norm(x)
    # boundary truncation loses only the ramp tails
    assert err < 0.4


def test_ewt_input_validation():
    with pytest.raises(ValueError):
        ewt(np.ones(30))
    with pytest.raises(ValueError):
        ewt(np.full(128, np.nan))
    with pytest.raises(ValueError):
        ewt(np.random.default_rng(0).standard_normal(128), n_bands=20)


def test_bench_empirical_wavelets():
    out = bench_empirical_wavelets()
    assert out["score"] == 1.0
    assert out["synthetic_ewt_leakage"] < 0.05
