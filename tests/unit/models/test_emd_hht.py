import numpy as np
import pytest

from quant_fund.models.emd_hht import bench_emd, emd, hilbert_spectrum, synth_emd


def test_bench_emd_passes():
    r = bench_emd()
    assert r["score"] == 1.0


def test_emd_reconstructs_input():
    x, _, _ = synth_emd(seed=7)
    r = emd(x)
    recon = np.sum(r["imfs"], axis=0) + r["resid"]
    assert np.max(np.abs(recon - x)) < 1e-8


def test_emd_imf_count_positive():
    x, _, _ = synth_emd(seed=3)
    r = emd(x)
    assert int(r["n_imf"][0]) >= 1


def test_emd_rejects_short_series():
    with pytest.raises(ValueError):
        emd(np.arange(10.0))


def test_hilbert_spectrum_shapes():
    x, _, _ = synth_emd(seed=1)
    r = hilbert_spectrum(x, max_imf=4)
    assert r["inst_freq"].shape == r["imfs"].shape if "imfs" in r else True
    assert r["dom_freqs"].size >= 1
