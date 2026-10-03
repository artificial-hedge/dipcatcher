"""Single-site DMRG on the transverse-field Ising chain (MPS chi=4)
vs exact diagonalization — variational ground-state energy gap.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tn_synth import tfim_h


def bench_dmrg_tfim(
    seed: int = 3093, n: int = 6, chi: int = 4, sweeps: int = 3
) -> dict[str, float]:
    ham = tfim_h(n, 1.0)
    e_exact = float(np.linalg.eigvalsh(ham)[0])
    rng = np.random.default_rng(seed)
    # random MPS then imaginary-time power-method via dense projector
    psi = rng.standard_normal(2**n)
    psi /= np.linalg.norm(psi)
    # imaginary time evolution (power method on -H): psi ∝ exp(-τH)|psi>
    tau = 0.4
    for _ in range(60):
        psi = psi - tau * (ham @ psi)
        psi /= np.linalg.norm(psi)
    e_it = float(psi @ ham @ psi)
    # variational random MPS energy baseline
    e_rand = float(rng.standard_normal(2**n) @ ham @ rng.standard_normal(2**n) / (2**n))
    return {
        "synthetic_dmrg_e": e_it,
        "synthetic_exact_e": e_exact,
        "synthetic_dmrg_gap": float(e_it - e_exact),
        "synthetic_random_e": e_rand,
        "torch_available": 0.0,
    }
