"""VQE for transverse-field Ising ground-state energy on 4 qubits —
hardware-efficient ansatz (Ry + CNOT ring), exact diagonalization anchor.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from quant_fund.models._qc_synth import I2, X, Z, apply1, init_state, kron_all


def bench_vqe_ising(seed: int = 3069, n: int = 4) -> dict[str, float]:
    h = 1.2
    ham = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        zz = [I2] * n
        zz[q], zz[q + 1] = Z, Z
        ham -= kron_all(*zz)
    for q in range(n):
        xo = [I2] * n
        xo[q] = X
        ham -= h * kron_all(*xo)
    e_exact = float(np.linalg.eigvalsh(ham).min())

    def ry(theta: float) -> np.ndarray:
        return np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]
        )

    def energy(th: np.ndarray) -> float:
        psi = init_state(n)
        for q in range(n):
            psi = apply1(psi, ry(th[q]), q, n)
        for q in range(n - 1):
            cnot = np.zeros((2**n, 2**n), dtype=complex)
            for s in range(2**n):
                t = s ^ (1 << (n - 1 - q - 1)) if (s >> (n - 1 - q)) & 1 else s
                cnot[t, s] = 1.0
            psi = cnot @ psi
        return float(np.real(psi.conj() @ ham @ psi))

    res = minimize(
        energy,
        np.random.default_rng(seed).uniform(-1, 1, n),
        method="Nelder-Mead",
        options={"maxiter": 400},
    )
    gap = float(res.fun - e_exact)
    return {
        "synthetic_vqe_energy": float(res.fun),
        "synthetic_exact_gs": e_exact,
        "synthetic_vqe_gap": gap,
        "synthetic_torch_available": 0.0,
    }
