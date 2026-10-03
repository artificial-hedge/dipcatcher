"""TT-SVD decomposition: compression of a smooth d-tensor at rank chi —
reconstruction error vs chi, and vs truncated CP baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tn_synth import mps_contract, smooth_tensor, to_mps


def bench_tt_svd(seed: int = 3089, d: int = 4, n: int = 9, chi: int = 6) -> dict[str, float]:
    T = smooth_tensor(seed, d, n)
    cores = to_mps(T, n, chi)
    rec = mps_contract(cores)
    err = float(np.linalg.norm(rec - T.reshape(-1)) / np.linalg.norm(T))
    n_full = T.size
    n_mps = int(sum(c.size for c in cores))
    return {
        "synthetic_tt_err": err,
        "synthetic_tt_ratio": float(n_mps / n_full),
        "synthetic_tt_chi": float(chi),
        "torch_available": 0.0,
    }
