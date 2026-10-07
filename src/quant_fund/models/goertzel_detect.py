"""Goertzel single-bin DFT detection vs a full FFT (SYNTHETIC).

The Goertzel recurrence computes one DFT bin in O(N) with O(1) state;
bench checks it matches rfft at the DTMF bins (941 + 1336 Hz) and that
the two-tone detector picks exactly the planted pair.
"""

import numpy as np

from quant_fund.models._sig3_synth import FS, two_tone


def _goertzel(x: np.ndarray, f: float) -> float:
    n = len(x)
    w = 2 * np.pi * f / FS
    cw = np.cos(w)
    s0 = s1 = s2 = 0.0
    for v in x:
        s0 = v + 2 * cw * s1 - s2
        s2 = s1
        s1 = s0
    return float(np.sqrt(s1 * s1 + s2 * s2 - 2 * cw * s1 * s2)) / n


def bench_goertzel_detect(seed: int = 4811) -> dict[str, float]:
    x = two_tone(seed)
    fft = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    g941 = _goertzel(x, 941.0)
    g1336 = _goertzel(x, 1336.0)
    f941 = float(fft[np.argmin(np.abs(freqs - 941.0))])
    f1336 = float(fft[np.argmin(np.abs(freqs - 1336.0))])
    # detector: top-2 Goertzel energies over DTMF grid
    dtmf = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
    e = {f: _goertzel(x, float(f)) for f in dtmf}
    top2 = sorted(e, key=lambda f: e[f], reverse=True)[:2]
    return {
        "synthetic_goe_941_err": abs(g941 - f941),
        "synthetic_goe_1336_err": abs(g1336 - f1336),
        "synthetic_goe_941": g941,
        "synthetic_goe_1336": g1336,
        "synthetic_goe_top1": float(top2[0]),
        "synthetic_goe_top2": float(top2[1]),
    }
