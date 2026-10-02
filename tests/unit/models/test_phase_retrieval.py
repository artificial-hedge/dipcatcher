import numpy as np

from quant_fund.models.phase_retrieval import (
    _align_corr,
    _cdi_measure,
    bench_phase_retrieval,
    gerchberg_saxton,
    wirtinger_flow,
)


def test_gs_recovers_spectrum():
    rng = np.random.default_rng(0)
    n = 16
    x = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    mag = np.abs(np.fft.fft(np.r_[x, np.zeros(n)]))
    z = gerchberg_saxton(mag, n, it=400, seed=0)
    mag_hat = np.abs(np.fft.fft(np.r_[z, np.zeros(n)]))
    assert np.linalg.norm(mag_hat - mag) / np.linalg.norm(mag) < 0.3


def test_wf_recovers_signal():
    rng = np.random.default_rng(1)
    n = 20
    x = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    masks = (rng.integers(0, 2, (6, n)) * 2 - 1).astype(np.complex128)
    y = _cdi_measure(x, masks)
    z = wirtinger_flow(masks, y, it=300, lr=1.0)
    assert _align_corr(z, x) > 0.9


def test_align_corr_scale_phase_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 8) + 1j * rng.normal(0, 1, 8)
    z = x * 3.0 * np.exp(1j * 0.7)
    assert _align_corr(z, x) > 0.999


def test_bench_phase_retrieval():
    out = bench_phase_retrieval(seed=550)
    assert out["synthetic_gs_spec_relerr"] < 0.2
    assert out["synthetic_wf_corr"] > 0.8
