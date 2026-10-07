"""Real cepstrum pitch detection on a harmonic train (SYNTHETIC).

cepstrum = IFFT(log |FFT(x)|^2); the pitch peak sits at quefrency
1/f0. Bench: detected f0 vs the planted 120 Hz fundamental, and the
harmonic comb's second-peak structure.
"""

import numpy as np

from quant_fund.models._sig3_synth import FS, voiced


def bench_cepstrum_pitch(seed: int = 4803, f0: float = 120.0) -> dict[str, float]:
    x = voiced(seed, f0)
    spec = np.abs(np.fft.rfft(x)) ** 2
    cep = np.fft.irfft(np.log(spec + 1e-12))
    q = np.arange(len(cep)) / FS  # quefrency axis
    lo, hi = int(FS / 300.0), int(FS / 60.0)  # 60..300 Hz range
    peak = lo + int(np.argmax(cep[lo:hi]))
    f0_hat = FS * 0 + 1.0 / q[peak]
    # second-order comb: cepstrum at 2*quefrency should also be elevated
    comb2 = cep[min(2 * peak, len(cep) - 1)]
    return {
        "synthetic_cep_f0": float(f0_hat),
        "synthetic_cep_f0_true": f0,
        "synthetic_cep_err": abs(float(f0_hat) - f0),
        "synthetic_cep_peak": float(cep[peak]),
        "synthetic_cep_comb2": float(comb2),
    }
