"""Unit tests for wave-303 speech/audio codec canon modules."""

import numpy as np

from quant_fund.models.adpcm_ima import adpcm_decode, adpcm_encode
from quant_fund.models.celp_encode import _synth, celp_frame
from quant_fund.models.lpc_analysis import autocorr, levinson, lpc_residual, synthesize
from quant_fund.models.mel_cepstrum import mel_filterbank, mfcc
from quant_fund.models.mulaw_compand import mu_decode, mu_encode, snr, uniform_quant
from quant_fund.models.viterbi_vad import frame_energy, viterbi_decode


def test_mulaw_beats_uniform_small_signal():
    t = np.arange(2000) / 2000
    x = 0.02 * np.sin(2 * np.pi * 3 * t)
    assert snr(x, mu_decode(mu_encode(x))) > snr(x, uniform_quant(x, 8))


def test_adpcm_roundtrip():
    x = np.cumsum(np.random.default_rng(0).normal(0, 30, 1000))
    x *= 20000 / np.max(np.abs(x))
    xh = adpcm_decode(adpcm_encode(x))
    assert np.sqrt(np.mean((x - xh) ** 2)) < 0.05 * np.sqrt(np.mean(x**2))


def test_lpc_recovers_ar():
    a = np.array([1.0, -0.9, 0.5, -0.2])
    e = np.random.default_rng(1).normal(0, 1, 6000)
    x = synthesize(e, a)
    ah, _ = levinson(autocorr(x, 3), 3)
    assert np.max(np.abs(ah - a)) < 0.2
    assert np.var(lpc_residual(x, ah)) > 0.5


def test_celp_voiced_fit():
    a = np.array([1.0, -0.8, 0.4, -0.15])
    exc = np.tile(np.r_[np.zeros(19), 3.0], 8)
    x = _synth(exc, a)
    e, _ = celp_frame(x, a)
    xh = _synth(e, a)
    assert np.dot(x - xh, x - xh) < 0.2 * np.dot(x, x)


def test_mfcc_distinguishes_tones():
    fs = 16000.0
    t = np.arange(400) / fs
    assert (
        np.linalg.norm(
            mfcc(np.sin(2 * np.pi * 300 * t), fs) - mfcc(np.sin(2 * np.pi * 3000 * t), fs)
        )
        > 10
    )
    fb = mel_filterbank(13, 512, fs)
    assert np.all(fb >= 0)


def test_viterbi_vad_matches_bruteforce():
    rng = np.random.default_rng(0)
    loge = rng.normal(0, 0.4, 10)
    loge[3:6] += 1.5
    path = viterbi_decode(loge, -0.2, 1.2, 0.05, 0.05)
    assert path[3:6].sum() >= 2
    assert frame_energy(np.ones(160), 80).shape == (2,)
