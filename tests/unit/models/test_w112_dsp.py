"""Tests for wave-112 DSP canon: remez, iir_design, biquad, filtfilt,
resample_poly, farrow."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.biquad import (
    bench_biquad,
    biquad_apply,
    biquad_lowpass,
    biquad_notch,
    biquad_peaking,
    biquad_response,
)
from quant_fund.models.farrow import bench_farrow, farrow_delay, farrow_lagrange
from quant_fund.models.filtfilt import bench_filtfilt, filtfilt
from quant_fund.models.iir_design import (
    bench_iir_design,
    iir_design,
    sos_response,
)
from quant_fund.models.remez import bench_remez, freq_response, remez
from quant_fund.models.resample_poly import bench_resample_poly, resample_poly


def test_remez_lowpass_spec():
    h = remez(41, [0.0, 0.1, 0.15, 0.5], [1.0, 0.0], [1.0, 10.0])
    assert np.abs(h - h[::-1]).max() < 1e-12
    assert np.abs(freq_response(h, np.linspace(0, 0.1, 32)) - 1).max() < 0.12
    assert freq_response(h, np.linspace(0.15, 0.5, 64)).max() < 0.05


def test_remez_highpass():
    h = remez(31, [0.0, 0.2, 0.25, 0.5], [0.0, 1.0], [10.0, 1.0])
    assert freq_response(h, np.array([0.45]))[0] > 0.9
    assert freq_response(h, np.array([0.05]))[0] < 0.05


def test_iir_butter_cutoff():
    sos = iir_design(5, 0.2, "butter")
    fc = sos_response(sos, np.array([0.2]))[0]
    assert abs(fc - 1 / np.sqrt(2)) < 0.02
    assert sos_response(sos, np.array([0.0]))[0] > 0.99
    assert sos_response(sos, np.array([0.45]))[0] < 0.001


def test_iir_stable_poles():
    for kind in ("butter", "cheby1"):
        sos = iir_design(4, 0.25, kind)
        for sec in sos:
            assert np.abs(np.roots([1.0, sec[4], sec[5]])).max() < 1.0


def test_biquad_cookbook():
    assert abs(biquad_response(biquad_lowpass(0.1), np.array([0.0]))[0] - 1) < 1e-9
    assert biquad_response(biquad_notch(0.2, q=16), np.array([0.2]))[0] < 1e-6
    g = biquad_response(biquad_peaking(0.15, 6.0, q=2.0), np.array([0.15]))[0]
    assert abs(20 * np.log10(g) - 6.0) < 0.1


def test_biquad_apply_finite():
    x = np.random.default_rng(0).normal(size=256)
    assert np.isfinite(biquad_apply(biquad_lowpass(0.1), x)).all()


def test_filtfilt_zero_phase():
    sos = iir_design(4, 0.1, "butter")
    t = np.arange(1024)
    x = np.sin(2 * np.pi * 0.05 * t)
    y = x.copy()
    for sec in sos:
        y = filtfilt(sec[:3], sec[3:], y)
    lag = int(np.argmax(np.correlate(y, x, "full")) - (len(t) - 1))
    assert abs(lag) <= 1


def test_resample_poly_lengths():
    x = np.sin(2 * np.pi * 0.05 * np.arange(600))
    assert abs(len(resample_poly(x, 3, 2)) - 900) <= 1
    assert abs(len(resample_poly(x, 2, 3)) - 400) <= 1


def test_resample_poly_fidelity():
    t = np.arange(600)
    x = np.sin(2 * np.pi * 0.04 * t)
    y = resample_poly(x, 3, 2)
    t2 = np.arange(len(y)) * 2 / 3
    err = np.abs(y[40:-40] - np.sin(2 * np.pi * 0.04 * t2[40:-40])).max()
    assert err < 0.01


def test_farrow_delay_accuracy():
    t = np.arange(256)
    x = np.sin(2 * np.pi * 0.05 * t)
    for mu in (0.25, 0.5, 0.75):
        y = farrow_delay(x, mu)
        ref = np.sin(2 * np.pi * 0.05 * (t - mu))
        assert np.abs(y[16:-16] - ref[16:-16]).max() < 0.02


def test_farrow_bank_shape():
    C = farrow_lagrange(3)
    assert C.shape == (4, 4)
    # unity DC at mu=0 → row eval at 0 gives tap m=0 only
    assert C[0, 0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "fn",
    [
        bench_remez,
        bench_iir_design,
        bench_biquad,
        bench_filtfilt,
        bench_resample_poly,
        bench_farrow,
    ],
    ids=lambda f: f.__name__,
)
def test_w112_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
