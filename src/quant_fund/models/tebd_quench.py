"""TEBD-style imaginary-time + real-time evolution on TFIM via
two-site gate decomposition (dense 6-qubit check vs exact expm).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from quant_fund.models._tn_synth import tfim_h


def bench_tebd_quench(
    seed: int = 3101, n: int = 5, t: float = 1.0, steps: int = 8
) -> dict[str, float]:
    ham = tfim_h(n, 1.0)
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi0 /= np.linalg.norm(psi0)
    # exact
    psi_exact = expm(-1j * ham * t) @ psi0
    # TEBD: split into even/odd bond terms + field; Strang splitting
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    ID = np.eye(2)

    def bond_op(q: int) -> np.ndarray:
        ops = [ID] * n
        ops[q], ops[q + 1] = Z, Z
        op: np.ndarray = ops[0]
        for o in ops[1:]:
            op = np.kron(op, o)
        return np.asarray(op)

    def field_op(q: int) -> np.ndarray:
        ops = [ID] * n
        ops[q] = X
        op: np.ndarray = ops[0]
        for o in ops[1:]:
            op = np.kron(op, o)
        return np.asarray(op)

    h_even = sum(-bond_op(q) for q in range(0, n - 1, 2))
    h_odd = sum(-bond_op(q) for q in range(1, n - 1, 2))
    h_field = -1.0 * sum(field_op(q) for q in range(n))
    dt = t / steps
    U1 = expm(-1j * h_even * dt / 2)
    U2 = expm(-1j * h_odd * dt / 2)
    U3 = expm(-1j * h_field * dt)
    psi = psi0.copy()
    for _ in range(steps):
        psi = U1 @ (U2 @ (U3 @ (U2 @ (U1 @ psi))))
    fid = float(abs(np.vdot(psi_exact, psi)) ** 2)
    return {
        "synthetic_tebd_fid": fid,
        "synthetic_tebd_err": float(1 - fid),
        "synthetic_tebd_dt": float(dt),
        "synthetic_torch_available": 0.0,
    }
