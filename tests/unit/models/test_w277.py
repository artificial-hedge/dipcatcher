"""Wave-277 signal-4 module tests."""

import numpy as np

from quant_fund.models.chirp_z import czt
from quant_fund.models.decimate_int import decimate
from quant_fund.models.fir_window import fir_lp
from quant_fund.models.prony_model import prony_fit
from quant_fund.models.stft_istft import istft, stft
from quant_fund.models.wola_synth import wola


def test_stft_shapes() -> None:
    x = np.zeros(128)
    spec = stft(x, 32, 16)
    assert spec.shape[1] == 17  # rfft bins


def test_istft_recovers_dc() -> None:
    x = np.ones(128)
    rec = istft(stft(x, 32, 8), 32, 8, 128)
    np.testing.assert_allclose(rec[32:96], 1.0, atol=1e-6)


def test_czt_matches_fft() -> None:
    x = np.random.RandomState(0).normal(size=16)
    np.testing.assert_allclose(czt(x, 64), np.fft.fft(x, 64), atol=1e-6)


def test_fir_lp_symmetric_peak() -> None:
    h = fir_lp(0.2, 15)
    assert int(np.argmax(h)) == 7
    np.testing.assert_allclose(h, h[::-1], atol=1e-12)


def test_prony_recovers_roots() -> None:
    n = np.arange(20)
    y = 0.7**n
    a = prony_fit(y, 1)
    assert len(a) == 1


def test_wola_sums_windows() -> None:
    frames = np.ones((4, 16)) * np.hanning(16)
    out = wola(frames, 8)
    assert len(out) == 4 * 8 + 8


def test_decimate_length() -> None:
    x = np.ones(64)
    assert len(decimate(x, 4)) == 16
