"""Wave-308 geophysics-2 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gardner_relation import fit_gardner, gardner_rho
from quant_fund.models.gassmann_sub import gassmann_dry, gassmann_forward, gassmann_sub
from quant_fund.models.reflectivity_synth import (
    impedance_to_reflectivity,
    ricker,
    synthetic_trace,
)
from quant_fund.models.semblance_scan import hyperbolic_gather
from quant_fund.models.spectral_decomp import first_notch_freq, thin_bed_trace
from quant_fund.models.vz_raytrace import ray_parameter, trace_ray, turning_depth


def test_reflectivity_sign_and_conv() -> None:
    z = np.array([2000.0, 2400.0, 2200.0])
    r = impedance_to_reflectivity(z)
    assert r[0] > 0 and r[1] < 0
    w = ricker(30.0, 0.001)
    tr = synthetic_trace(r, w)
    assert np.allclose(tr, np.convolve(r, w))


def test_gassmann_roundtrip_and_monotone() -> None:
    kd = gassmann_dry(gassmann_forward(7.0, 37.0, 2.2, 0.25), 37.0, 2.2, 0.25)
    assert abs(kd - 7.0) < 1e-6
    k2, mu2 = gassmann_sub(10.0, 5.0, 37.0, 2.2, 0.14, 0.25)
    assert k2 < 10.0 and mu2 == 5.0


def test_spectral_notch_thickness() -> None:
    dt = 0.001
    h = int(np.ceil(1.5 / (30.0 * dt)))
    t = np.arange(-h, h + 1) * dt
    a = np.pi**2 * 30.0**2 * t**2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    tr = thin_bed_trace(0.4, 15, 400, w)
    f_n = first_notch_freq(tr, dt)
    assert abs(1.0 / f_n / dt - 15) <= 2


def test_semblance_gather_shape() -> None:
    offs = np.linspace(50.0, 1500.0, 20)
    g = hyperbolic_gather([(0.4, 2200.0, 1.0)], offs, 0.002, 400, np.ones(5))
    assert g.shape == (400, 20)
    assert np.abs(g).max() > 0


def test_gardner_fit_recovers_params() -> None:
    v = np.linspace(1500.0, 4500.0, 200)
    a, b = fit_gardner(v, gardner_rho(v))
    assert abs(a - 0.31) < 1e-6 and abs(b - 0.25) < 1e-6


def test_vz_raytrace_snell_and_turning() -> None:
    v0, k, th0 = 1800.0, 0.6, 0.6
    x, z, t, th = trace_ray(v0, k, th0)
    p = ray_parameter(v0, th0)
    assert np.max(np.abs(np.sin(th) / (v0 + k * z) - p)) < 1e-9
    assert abs(z[-1] - turning_depth(v0, k, th0)) < 1e-3


def test_vz_time_monotone() -> None:
    _, _, t, _ = trace_ray(1800.0, 0.6, 0.5)
    assert np.all(np.diff(t) > 0)
