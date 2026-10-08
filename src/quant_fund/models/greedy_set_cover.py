"""Greedy set cover — the ln(n)-approximation (SYNTHETIC).

Picks the set minimizing cost-per-newly-covered-element until the
universe is covered; classic Chvátal bound H(m). Bench compares the
greedy cost against the brute-force optimum on the shared cover
fixture and the LP-style lower bound.
"""

import numpy as np

from quant_fund.models._ilp_synth import SC_A, SC_C, brute_set_cover


def _greedy() -> tuple[float, int]:
    m, n = SC_A.shape
    covered = np.zeros(SC_A.shape[0], dtype=bool)
    total = 0.0
    chosen = 0
    while not covered.all():
        gains = SC_A[~covered, :].sum(axis=0)
        if gains.max() <= 0:
            break
        # argmin c_j / gain_j
        ratio = np.divide(SC_C, gains, out=np.full_like(SC_C, np.inf), where=gains > 0)
        j = int(np.argmin(ratio))
        covered |= SC_A[:, j] > 0
        total += SC_C[j]
        chosen += 1
    return total, chosen


def bench_greedy_set_cover(seed: int = 5201) -> dict[str, float]:
    cost, chosen = _greedy()
    truth = brute_set_cover(SC_C, SC_A)
    return {
        "synthetic_gsc_cost": cost,
        "synthetic_gsc_sets": float(chosen),
        "synthetic_gsc_truth": truth,
        "synthetic_gsc_ratio": cost / truth,
        "synthetic_gsc_lnm_bound": float(np.log(SC_A.shape[0]) + 1.0),
    }
