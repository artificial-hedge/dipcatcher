import numpy as np
import pytest

from quant_fund.models.vmd import bench_vmd, mode_dominant_freq, vmd


def _signal(rng):
    t = np.arange(512)
    return (
        np.cos(2 * np.pi * 0.04 * t)
        + 0.7 * np.cos(2 * np.pi * 0.12 * t + 0.6)
        + 0.5 * np.cos(2 * np.pi * 0.22 * t + 1.1)
        + 0.05 * rng.standard_normal(t.size)
    )


def test_vmd_shapes():
    rng = np.random.default_rng(0)
    out = vmd(_signal(rng), n_modes=3, n_iter=100)
    assert out["modes"].shape == (3, 512)
    assert out["center_freqs"].shape == (3,)
    assert out["spectra"].shape[0] == 3


def test_vmd_centers_on_tones():
    rng = np.random.default_rng(1)
    out = vmd(_signal(rng), n_modes=3, n_iter=400)
    centers = np.asarray(out["center_freqs"])
    assert np.abs(centers - np.array([0.04, 0.12, 0.22])).max() < 0.03


def test_vmd_reconstruction():
    rng = np.random.default_rng(2)
    f = _signal(rng)
    out = vmd(f, n_modes=3, n_iter=400)
    err = np.linalg.norm(f - np.asarray(out["modes"]).sum(axis=0)) / np.linalg.norm(f)
    assert err < 0.2


def test_mode_dominant_freq():
    rng = np.random.default_rng(3)
    out = vmd(_signal(rng), n_modes=3, n_iter=300)
    dom = mode_dominant_freq(np.asarray(out["modes"]), np.asarray(out["spectra"]))
    assert dom.shape == (3,)
    assert np.all(dom > 0)


def test_vmd_input_validation():
    with pytest.raises(ValueError):
        vmd(np.arange(5.0))
    with pytest.raises(ValueError):
        vmd(np.full(100, np.nan))
    with pytest.raises(ValueError):
        vmd(np.random.default_rng(0).standard_normal(100), n_modes=0)


def test_bench_vmd():
    out = bench_vmd()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_vmd_freq_err"] < 0.03
