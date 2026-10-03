"""QAOA (Farhi 2014) p=1 MaxCut on a 4-qubit random graph — statevector
sim, expected cut vs optimal and vs random uniform sampling.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from quant_fund.models._qc_synth import (
    I2,
    H,
    X,
    Z,
    apply1,
    init_state,
    maxcut_cost,
    maxcut_graph,
    measure_probs,
)


def bench_qaoa_maxcut(seed: int = 3065) -> dict[str, float]:
    n = 4
    edges = maxcut_graph(seed, n)
    opt = max(maxcut_cost(x, edges) for x in range(2**n))

    def qaoa(params: np.ndarray) -> float:
        g, b = params
        psi = init_state(n)
        for q in range(n):
            psi = apply1(psi, H, q, n)
        # cost unitary: exp(-i g C) where C = sum ZZ terms on edges
        for i, j in edges:
            zz_vals = np.ones(1)
            for q in range(n):
                zz_vals = np.kron(zz_vals, np.diag(Z) if q in (i, j) else np.diag(I2))
            psi = np.exp(-0.5j * g * (1 - zz_vals)) * psi
        for q in range(n):
            psi = apply1(psi, np.cos(b) * I2 - 1j * np.sin(b) * X, q, n)
        probs = measure_probs(psi)
        return -float(sum(probs[x] * maxcut_cost(x, edges) for x in range(2**n)))

    res = minimize(qaoa, np.array([0.3, 0.4]), method="Nelder-Mead", options={"maxiter": 60})
    exp_cut = -res.fun
    uniform = float(np.mean([maxcut_cost(x, edges) for x in range(2**n)]))
    return {
        "synthetic_qaoa_cut": exp_cut,
        "synthetic_maxcut_opt": float(opt),
        "synthetic_uniform_cut": uniform,
        "synthetic_qaoa_ratio": float(exp_cut / max(opt, 1e-9)),
        "synthetic_qaoa_gain": float(exp_cut - uniform),
        "torch_available": 0.0,
    }
