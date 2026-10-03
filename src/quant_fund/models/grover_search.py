"""Grover amplitude amplification on 4 qubits: marked-state probability
after optimal iterations vs O(N) classical expected queries.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._qc_synth import H, apply1, init_state, measure_probs


def bench_grover_search(seed: int = 3073, n: int = 4, marked: int = 5) -> dict[str, float]:
    N = 2**n
    psi = init_state(n)
    for q in range(n):
        psi = apply1(psi, H, q, n)
    k_opt = int(np.floor(np.pi / 4 * np.sqrt(N)))
    p_series = []
    for _ in range(k_opt):
        # oracle: flip phase of marked state
        psi[marked] *= -1
        # diffusion: 2|s><s| - I
        psi = 2 * np.full(N, psi.mean(), dtype=complex) - psi
        p_series.append(float(measure_probs(psi)[marked]))
    p_final = float(measure_probs(psi)[marked])
    classical = float(N) / 2  # expected queries for classical search
    return {
        "synthetic_grover_iters": float(k_opt),
        "synthetic_grover_p_marked": p_final,
        "synthetic_grover_classical_queries": classical,
        "synthetic_grover_speedup": float(classical / k_opt),
        "torch_available": 0.0,
    }
