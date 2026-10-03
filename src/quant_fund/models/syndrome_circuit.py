"""Stabilizer-syndrome measurement circuits on a small statevector.

Simulates ancilla-assisted stabilizer measurement at gate level: for a Z-type
check, CNOTs from each data qubit in the check support into the ancilla then
Z-measure; for an X-type check, Hadamard-sandwiched CNOTs ancilla->data
(phase kickback). Verified: on the Bell state |00>+|11>, injecting a Pauli
error flips exactly the expected syndrome bits, and the clean circuit
leaves the state untouched (fault-free ancilla disentangles).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 948


def _apply_1q(psi: np.ndarray, n: int, q: int, gate: np.ndarray) -> np.ndarray:
    """Apply 1-qubit gate on qubit q (q=0 -> MSB of index)."""
    dim = 1 << n
    mask = 1 << (n - 1 - q)
    out = np.zeros_like(psi)
    for i in range(dim):
        b = (i & mask) >> (n - 1 - q)
        for bb in (0, 1):
            j = (i & ~mask) | (bb << (n - 1 - q))
            out[j] += gate[bb, b] * psi[i]
    return out


def _apply_cnot(psi: np.ndarray, n: int, c: int, t: int) -> np.ndarray:
    dim = 1 << n
    cm = 1 << (n - 1 - c)
    tm = 1 << (n - 1 - t)
    out = np.zeros_like(psi)
    for i in range(dim):
        j = i ^ tm if i & cm else i
        out[j] += psi[i]
    return out


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def bell_state(n_data: int = 2, n_anc: int = 2) -> np.ndarray:
    """|00+11>/sqrt2 on data qubits 0,1, ancillas |00>."""
    n = n_data + n_anc
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    psi = _apply_1q(psi, n, 0, _H)
    psi = _apply_cnot(psi, n, 0, 1)
    return psi


def z_check(psi: np.ndarray, n: int, anc: int, support: list[int]) -> np.ndarray:
    """Measure parity of `support` into ancilla `anc` (Z-type check)."""
    for q in support:
        psi = _apply_cnot(psi, n, q, anc)
    return psi


def x_check(psi: np.ndarray, n: int, anc: int, support: list[int]) -> np.ndarray:
    """X-type check via ancilla phase kickback."""
    psi = _apply_1q(psi, n, anc, _H)
    for q in support:
        psi = _apply_cnot(psi, n, anc, q)
    psi = _apply_1q(psi, n, anc, _H)
    return psi


def measure_qubit(
    psi: np.ndarray, n: int, q: int, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
    """Z-measure qubit q; returns (outcome, collapsed renormalized state)."""
    dim = 1 << n
    mask = 1 << (n - 1 - q)
    p1 = float(np.sum(np.abs(psi[mask & np.arange(dim) != 0]) ** 2))
    out = int(rng.random() < p1)
    collapsed = psi.copy()
    keep = mask if out else 0
    for i in range(dim):
        if (i & mask) != keep:
            collapsed[i] = 0.0
    nrm = np.linalg.norm(collapsed)
    return out, collapsed / nrm if nrm > 0 else collapsed


def bench_syndrome_circuit(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 4
    checks = []
    # clean run: syndrome (ZZ on data{0,1} -> anc2, XX on data{0,1} -> anc3)
    psi = bell_state()
    psi = z_check(psi, n, 2, [0, 1])
    psi = x_check(psi, n, 3, [0, 1])
    o2, psi = measure_qubit(psi, n, 2, rng)
    o3, psi = measure_qubit(psi, n, 3, rng)
    checks.append(o2 == 0 and o3 == 0)
    # state after clean measurement is still the Bell state on data
    bell = np.zeros(1 << n, dtype=np.complex128)
    bell[0] = 1 / np.sqrt(2)
    bell[0b1100] = 1 / np.sqrt(2)
    checks.append(abs(np.vdot(bell, psi)) ** 2 > 0.999)
    # X error on data qubit 0: Z-check fires (1), X-check quiet (0)
    psi = bell_state()
    psi = _apply_1q(psi, n, 0, _X)
    psi = z_check(psi, n, 2, [0, 1])
    psi = x_check(psi, n, 3, [0, 1])
    o2, psi = measure_qubit(psi, n, 2, rng)
    o3, psi = measure_qubit(psi, n, 3, rng)
    checks.append(o2 == 1 and o3 == 0)
    # Z error: X-check fires, Z-check quiet
    psi = bell_state()
    psi = _apply_1q(psi, n, 0, _Z)
    psi = z_check(psi, n, 2, [0, 1])
    psi = x_check(psi, n, 3, [0, 1])
    o2, psi = measure_qubit(psi, n, 2, rng)
    o3, psi = measure_qubit(psi, n, 3, rng)
    checks.append(o2 == 0 and o3 == 1)
    return {"synthetic_syndrome_circuit": float(np.mean(checks))}
