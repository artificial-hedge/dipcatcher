"""MPS round-trip fidelity: compress an arbitrary dense state to
chi-bond MPS, measure retained fidelity vs chi.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tn_synth import mps_contract, to_mps


def bench_mps_fidelity(seed: int = 3105, n: int = 6, chi: int = 4) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n)
    psi /= np.linalg.norm(psi)
    T = psi.reshape([2] * n)
    cores = to_mps(T, 2, chi)
    rec = mps_contract(cores)
    rec /= np.linalg.norm(rec)
    fid = float(abs(np.vdot(rec, psi)) ** 2)
    # entangled GHZ state round-trip at chi=2 (exact)
    ghz = np.zeros(2**n)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    cores2 = to_mps(ghz.reshape([2] * n), 2, 2)
    rec2 = mps_contract(cores2)
    rec2 /= np.linalg.norm(rec2)
    fid_ghz = float(abs(np.vdot(rec2, ghz)) ** 2)
    return {
        "synthetic_mps_fid_rand": fid,
        "synthetic_mps_fid_ghz": fid_ghz,
        "synthetic_mps_chi": float(chi),
        "synthetic_torch_available": 0.0,
    }
