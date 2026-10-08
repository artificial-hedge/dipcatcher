"""MVDR (Capon) beamformer on a ULA snapshot matrix (SYNTHETIC).

w = R^{-1} a / (a^H R^{-1} a) minimizes output power subject to unit gain
in the look direction. Bench: beampattern peak at the planted DOA and
null depth toward the interferer vs a delay-sum (Bartlett) beamformer.
"""

import numpy as np

from quant_fund.models._sig3_synth import array_snap


def bench_mvdr_beamformer(seed: int = 4805, theta: float = 20.0) -> dict[str, float]:
    x, steer = array_snap(seed, theta)
    m = x.shape[0]
    r = x @ x.conj().T / x.shape[1] + 1e-6 * np.eye(m)
    ri = np.linalg.inv(r)
    a = steer
    w = ri @ a / (a.conj() @ ri @ a)
    # response at true DOA (constraint: 0 dB) and at the -35 deg
    # interferer (adaptive null) vs a delay-sum beamformer
    s_true = np.exp(-1j * np.pi * np.arange(m) * np.sin(np.deg2rad(theta)))
    s_int = np.exp(-1j * np.pi * np.arange(m) * np.sin(np.deg2rad(-35.0)))
    null_db = 20 * np.log10(abs(w.conj() @ s_int) + 1e-12)
    gain_db = 20 * np.log10(abs(w.conj() @ s_true) + 1e-12)
    w_ds = a / m
    ds_db = 20 * np.log10(abs(w_ds.conj() @ s_int) + 1e-12)
    return {
        "synthetic_mvdr_constraint_err": abs(abs(w.conj() @ a) - 1.0),
        "synthetic_mvdr_null_db": float(null_db),
        "synthetic_mvdr_gain_db": float(gain_db),
        "synthetic_mvdr_ds_null_db": float(ds_db),
        "synthetic_mvdr_null_gain": float(ds_db - null_db),
    }
