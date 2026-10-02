"""Gardner canon: Gardner timing-error detector with
interpolation + PI loop — recovers symbol timing offset on
2-samples/symbol RRC-shaped BPSK streams. Bench: static offset
estimated within a fraction of a symbol period and decisions
beat the unsynchronized slice. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.rrc_filter import matched_filter, pulse_shape, rrc_taps

FloatArray = NDArray[np.float64]


def _interp(x: FloatArray, pos: float) -> float:
    i = int(pos)
    f = pos - i
    i = min(max(i, 0), len(x) - 2)
    return float(x[i] + f * (x[i + 1] - x[i]))


def gardner_recover(
    rx: FloatArray,
    sps: int = 8,
    loop_gain: float = 0.04,
    integ_gain: float = 0.002,
) -> tuple[FloatArray, FloatArray]:
    """Gardner TED on an sps-oversampled stream: strobe at
    k*sps + mu with a PI-controlled fractional offset.

    Returns (symbol decisions, mu trace).
    """
    rx = np.asarray(rx, dtype=np.float64)
    n_sym = int(len(rx) / sps) - 2
    out = np.zeros(n_sym)
    mu_trace = np.zeros(n_sym)
    mu = 0.0
    v_int = 0.0
    prev = _interp(rx, mu)
    for k in range(1, n_sym):
        pos = k * sps + mu
        if pos >= len(rx) - 1:
            pos = len(rx) - 1.5
        x = _interp(rx, pos)
        x_mid = _interp(rx, pos - sps / 2)
        out[k] = 1.0 if x >= 0 else -1.0
        e = (x - prev) * x_mid
        v_int += integ_gain * e
        mu -= loop_gain * e + v_int
        mu = max(min(mu, sps), -sps)
        mu_trace[k] = mu
        prev = x
    return out, mu_trace


def bench_gardner(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    sps = 8
    n_sym = 300
    bits = (2 * rng.integers(0, 2, n_sym) - 1).astype(np.float64)
    # RRC-shaped baseband: ISI-free only at the right instant
    h = rrc_taps(sps, 8 * sps + 1, beta=0.35)
    tx = pulse_shape(bits, sps, h)
    mf = matched_filter(tx, h)
    off = 3  # samples ≈ 0.375 symbol — ISI appears off-center
    rx = mf[off:] + rng.normal(0, 0.05, len(mf[off:]))
    dec, mu = gardner_recover(rx[: (n_sym - 2) * sps], sps=sps)
    st = n_sym // 4
    out["synthetic_gardner_ber"] = float(np.mean(dec[st : len(dec)] != bits[st : len(dec)]))
    # unsynchronized: decode the same stream with mu pinned to 0
    dec_fixed, _ = gardner_recover(rx[: (n_sym - 2) * sps], sps=sps, loop_gain=0.0, integ_gain=0.0)
    out["synthetic_gardner_base_ber"] = float(
        np.mean(dec_fixed[st : len(dec)] != bits[st : len(dec)])
    )
    out["synthetic_gardner_gain"] = out["synthetic_gardner_base_ber"] - out["synthetic_gardner_ber"]
    # zero-offset control: aligned stream → low BER too
    rx0 = mf + rng.normal(0, 0.05, len(mf))
    dec0, _ = gardner_recover(rx0[: (n_sym - 2) * sps], sps=sps)
    out["synthetic_gardner_aligned_ber"] = float(
        np.mean(dec0[st : len(dec0)] != bits[st : len(dec0)])
    )
    out["synthetic_gardner_mu_std"] = float(np.std(mu[st:]))
    return out
