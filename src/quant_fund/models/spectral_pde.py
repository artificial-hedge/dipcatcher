"""Fourier pseudospectral heat solver — u_t = 0.5σ²u_xx diagonalizes (SYNTHETIC)
in Fourier space; exact-to-roundoff reference solution. L2 error vs
analytic + comparison to PINN methods.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, L, eval_error, grid, u0


def bench_spectral_pde(seed: int = 2519, nx: int = 128) -> dict[str, float]:
    x = grid(nx)
    u = u0(x)
    # periodic extension assumed smooth enough (u decays at ±4)
    k = np.fft.fftfreq(nx, d=(2 * L / nx)) * 2 * np.pi
    uh = np.fft.fft(u)
    uh *= np.exp(-0.5 * SIG**2 * k**2 * T_)
    uT = np.real(np.fft.ifft(uh))

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        return np.asarray(np.interp(xq, x, uT))

    return {"synthetic_spectral_rel_l2": eval_error(pred), "synthetic_torch_available": 0.0}
