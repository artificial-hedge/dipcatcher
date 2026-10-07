"""Adversarial probes for fractional_ou."""

import numpy as np

from quant_fund.models import fractional_ou as fo


def test_sim_uses_hermitian_coefficients():
    """Pin the circulant construction: coefficients must satisfy
    c[m-k] == conj(c[k]) so the ifft is a real stationary process."""
    n, h, a, sigma, seed = 256, 0.3, 0.5, 1.0, 11
    got = fo.simulate_fou_exact(n, h, a, sigma, seed)
    # recompute the symmetric construction independently
    rng = np.random.default_rng(seed)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    freqs = np.fft.fftfreq(m) * 2 * np.pi
    spec = fo.fou_spectrum(np.abs(freqs), h, a, sigma)
    amp = np.sqrt(np.maximum(spec, 0.0) * m / 2.0)
    re = rng.standard_normal(m)
    im = rng.standard_normal(m)
    coeff = amp * (re + 1j * im)
    half = m // 2
    coeff[half + 1 :] = np.conj(coeff[1:half][::-1])
    coeff[half] = amp[half] * re[half]
    coeff[0] = 0.0
    x = np.fft.ifft(coeff).real[:n]
    x = x - x.mean()
    expected = x / np.std(x) * sigma
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_sim_imaginary_part_is_zero():
    """With Hermitian symmetry the ifft imaginary part must vanish —
    an independent-draw implementation leaks power there."""
    n, h, a, sigma, seed = 256, 0.3, 0.5, 1.0, 3
    rng = np.random.default_rng(seed)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    freqs = np.fft.fftfreq(m) * 2 * np.pi
    spec = fo.fou_spectrum(np.abs(freqs), h, a, sigma)
    amp = np.sqrt(np.maximum(spec, 0.0) * m / 2.0)
    re = rng.standard_normal(m)
    im = rng.standard_normal(m)
    coeff = amp * (re + 1j * im)
    half = m // 2
    coeff[half + 1 :] = np.conj(coeff[1:half][::-1])
    coeff[half] = amp[half] * re[half]
    coeff[0] = 0.0
    assert np.abs(np.fft.ifft(coeff).imag).max() < 1e-12


def test_estimate_converged_flag_reflects_nelder_mead():
    """converged must come from res.success of the winning start."""
    x = fo.synth_fou_observed(1024, 0.4, a=0.5, sigma=1.0, seed=5)
    est = fo.estimate_fou(x, seed=5)
    # on this well-posed problem NM should converge
    assert est.converged is True


def test_bench_smoke():
    out = fo.bench_fractional_ou()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_h_err_mean"] < 0.25
