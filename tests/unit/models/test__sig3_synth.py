"""Probes for _sig3_synth (signal fixtures)."""

import numpy as np
import pytest

from quant_fund.models._sig3_synth import FS, N, array_snap, chirp, two_tone, voiced


def test_chirp_deterministic_and_energy():
    a = chirp(0)
    b = chirp(0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (N,)
    assert np.isfinite(a).all()
    # chirp sweeps 200→800 Hz: low and high band both carry energy
    sp = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    low = sp[(freqs >= 200) & (freqs < 350)].mean()
    high = sp[(freqs >= 650) & (freqs < 800)].mean()
    assert low > 0 and high > 0


def test_voiced_harmonic_peaks():
    s = voiced(0, f0=150.0)
    sp = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    peak = freqs[sp.argmax()]
    assert abs(peak - 150.0) < 3.0  # fundamental dominant


@pytest.mark.parametrize("f0", [0.0, -50.0, np.nan, FS])
def test_voiced_hostile_f0(f0):
    with pytest.raises(ValueError):
        voiced(0, f0=f0)


def test_two_tone_dtmf_peaks():
    s = two_tone(0)
    sp = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    top2 = freqs[np.argsort(sp)[-2:]]
    assert {round(f / 5) * 5 for f in top2} == {940, 1335} or all(
        min(abs(f - 941), abs(f - 1336)) < 5 for f in top2
    )


def test_array_snap_shapes_and_steering_norm():
    x, steer = array_snap(0, theta_deg=15.0)
    assert x.shape == (8, 256) and steer.shape == (8,)
    assert steer.dtype == np.complex128
    np.testing.assert_allclose(np.abs(steer), 1.0)
    assert np.isfinite(x.real).all() and np.isfinite(x.imag).all()


@pytest.mark.parametrize("th", [np.nan, np.inf, 100.0, -91.0])
def test_array_snap_hostile_theta(th):
    with pytest.raises(ValueError):
        array_snap(0, theta_deg=th)
