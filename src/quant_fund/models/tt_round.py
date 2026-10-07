"""TT rounding: compress a noisy low-TT-rank tensor back to small chi —
error vs naive truncation to CP-rank-1 baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tn_synth import mps_contract, smooth_tensor, to_mps


def bench_tt_round(seed: int = 3109, d: int = 4, n: int = 8, chi: int = 4) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    T = smooth_tensor(seed, d, n)
    Tn = T + 0.05 * rng.standard_normal(T.shape)
    # round at chi=4
    cores = to_mps(Tn, n, chi)
    rec = mps_contract(cores).reshape(T.shape)
    err_tt = float(np.linalg.norm(rec - T) / np.linalg.norm(T))
    # CP-1 baseline: best rank-1 approx via HOSVD lead
    err_rank1 = 1.0
    U1: list[np.ndarray] = []
    for k in range(d):
        M = Tn.transpose(k, *[i for i in range(d) if i != k]).reshape(n, -1)
        u = np.linalg.svd(M, full_matrices=False)[0][:, 0]
        U1.append(u)
    r1 = np.ones([n] * d)
    for k in range(d):
        shape = [1] * d
        shape[k] = n
        r1 = r1 * U1[k].reshape(shape)
    scale = float((Tn * r1).sum() / (r1**2).sum())
    err_rank1 = float(np.linalg.norm(scale * r1 - T) / np.linalg.norm(T))
    return {
        "synthetic_ttround_err": err_tt,
        "synthetic_rank1_err": err_rank1,
        "synthetic_ttround_gain": float(err_rank1 - err_tt),
        "synthetic_torch_available": 0.0,
    }
