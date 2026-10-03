"""Unit tests for wave-301 geophysics/seismic canon modules."""

import numpy as np

from quant_fund.models.avo_shuey import avo_class, fit_shuey, shuey
from quant_fund.models.eikonal_fmm import fast_march, head_wave_time
from quant_fund.models.kirchhoff_mig import diffractor_gather, kirchhoff_migrate
from quant_fund.models.nmo_dix import dix_interval, fit_vnmo, nmo_correct, nmo_curve
from quant_fund.models.taup_transform import slant_stack
from quant_fund.models.vibroseis_sweep import correlate, linear_sweep


def test_nmo_flatten():
    x = np.linspace(0, 1500, 20)
    t = nmo_curve(0.8, x, 2000.0)
    resid = nmo_correct(t, x, 2000.0) - 0.8
    assert np.max(np.abs(resid)) < 1e-9
    assert abs(fit_vnmo(0.8, x, t) - 2000.0) / 2000.0 < 0.01


def test_dix_monotonic():
    vrms = np.array([1500.0, 1800.0, 2100.0])
    t0s = np.array([0.5, 1.0, 1.5])
    vi = dix_interval(vrms, t0s)
    assert vi.shape == (3,)
    assert np.all(vi > 0)


def test_slant_stack_shape():
    d = np.zeros((100, 10))
    d[50, :] = 1.0
    out = slant_stack(d, np.arange(10) * 30.0, np.arange(100) * 0.004, np.array([0.0, 1e-4]))
    assert out.shape == (100, 2)
    assert np.argmax(out[:, 0]) == 50


def test_migrate_focus():
    x = np.arange(20) * 25.0
    t = np.arange(200) * 0.004
    v = 2000.0
    xd, zd = 250.0, 200.0
    data = np.zeros((200, 20))
    for k in range(20):
        it = int(round(diffractor_gather(x, v, xd, zd)[k] / 0.004))
        data[it, k] = 1.0
    img = kirchhoff_migrate(data, x, t, v)
    iz, ix = np.unravel_index(np.argmax(img), img.shape)
    assert abs(x[ix] - xd) < 60.0
    assert abs(v * t[iz] / 2 - zd) < 60.0


def test_shuey_roundtrip():
    th = np.linspace(0, 35, 20)
    r = shuey(th, 0.1, -0.2, 0.03)
    a, b, c = fit_shuey(th, r)
    assert abs(a - 0.1) < 1e-6 and abs(b + 0.2) < 1e-6 and abs(c - 0.03) < 1e-6
    assert avo_class(0.12, -0.1) == 1
    assert avo_class(-0.1, -0.2) == 3


def test_sweep_correlation():
    dt = 0.001
    t = np.arange(2000) * dt
    pilot = linear_sweep(t[:1000], 5.0, 80.0, 1.0)
    refl = np.zeros(2000)
    refl[800] = 1.0
    data = np.convolve(refl, pilot)[:2000]
    corr = correlate(data, pilot)
    assert np.argmax(np.abs(corr)) == 800 + 999


def test_eikonal_constant():
    nz = nx = 40
    slow = np.full((nz, nx), 1.0 / 2000.0)
    T = fast_march(slow, (5, 5), 5.0)
    zz, xx = np.mgrid[0:nz, 0:nx]
    d = np.sqrt(((zz - 5) * 5.0) ** 2 + ((xx - 5) * 5.0) ** 2)
    assert np.max(np.abs(T - d / 2000.0)[d > 15]) / np.max(d / 2000.0) < 0.03
    assert head_wave_time(700.0, 1500.0, 4000.0, 200.0) < 700.0 / 1500.0
