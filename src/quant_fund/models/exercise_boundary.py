"""Early-exercise boundary estimation from a coarse binomial tree (SYNTHETIC).

The boundary at each time step is the deepest spot price where exercise
still dominates continuation. Bench: L2 distance between the coarse
boundary and the 2000-step reference over the middle of the contract, plus
the naive always-at-K boundary as the honest-negative baseline.
"""

import numpy as np

from quant_fund.models._amopt_synth import K, crr_price


def bench_exercise_boundary(seed: int = 4111, n_coarse: int = 400) -> dict[str, float]:
    del seed
    ref_price, ref_bnd = crr_price(2000)
    _price, bnd = crr_price(n_coarse)
    lo, hi = n_coarse // 4, (3 * n_coarse) // 4
    rr = np.linspace(lo, hi, hi - lo) / n_coarse * 2000
    rr = np.clip(rr.astype(int), 0, 1999)
    ref_seg = ref_bnd[rr]
    seg = bnd[lo:hi]
    err = float(np.sqrt(np.mean((seg - ref_seg) ** 2)))
    naive = float(np.sqrt(np.mean((np.full_like(seg, K) - ref_seg) ** 2)))
    return {
        "synthetic_boundary_l2": err,
        "synthetic_boundary_naive_l2": naive,
        "synthetic_boundary_midpoint": float(seg[len(seg) // 2]),
        "synthetic_boundary_ref_midpoint": float(ref_seg[len(seg) // 2]),
    }
