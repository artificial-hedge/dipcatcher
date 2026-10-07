"""Quantum phase estimation on a controlled-U (Z-rotation) with a
planted phase — recovers binary digits vs uniform bit guessing.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._qc_synth import measure_probs


def bench_qpe_phase(seed: int = 3077, prec: int = 3) -> dict[str, float]:
    theta_true = 0.625  # planted phase in [0,1)
    N = 2**prec
    # register state: uniform superposition × eigenstate
    psi = np.ones(N, dtype=complex) / np.sqrt(N)
    # controlled-U^{2^j}: amplitudes pick up exp(2πi θ k)
    k = np.arange(N)
    psi = psi * np.exp(2j * np.pi * theta_true * k)
    # inverse QFT
    f = np.fft.fft(np.eye(N)) / np.sqrt(N)
    out = f @ psi
    probs = measure_probs(out)
    est = float(np.argmax(probs)) / N
    # randomized baseline: sample uniform
    rng = np.random.default_rng(seed)
    base_err = float(np.abs(rng.integers(0, N) / N - theta_true))
    return {
        "synthetic_qpe_est": est,
        "synthetic_qpe_true": theta_true,
        "synthetic_qpe_err": float(abs(est - theta_true)),
        "synthetic_qpe_random_err": base_err,
        "synthetic_qpe_peak_prob": float(probs.max()),
        "synthetic_torch_available": 0.0,
    }
