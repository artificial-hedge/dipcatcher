"""Shared fixture for wave-199 quantum canon — statevector helpers + (SYNTHETIC)
small planted-problem instances (all n<=4 qubits, classical sim).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def kron_all(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return np.asarray(out)


def init_state(n: int) -> ComplexArray:
    if n < 1:
        raise ValueError(f"need n>=1 qubits, got {n}")
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    return np.asarray(psi)


def apply1(psi: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    if n < 1 or not 0 <= qubit < n:
        raise ValueError(f"need 0 <= qubit < n with n>=1, got qubit={qubit}, n={n}")
    if np.asarray(gate).shape != (2, 2):
        raise ValueError(f"1-qubit gate must be (2,2), got {np.asarray(gate).shape}")
    if len(psi) != 2**n:
        raise ValueError(f"psi length {len(psi)} != 2**n for n={n}")
    ops = [I2] * n
    ops[qubit] = gate
    return np.asarray(kron_all(*ops) @ psi)


def measure_probs(psi: np.ndarray) -> FloatArray:
    return np.asarray(np.abs(psi) ** 2)


def maxcut_graph(seed: int, n: int = 4) -> list[tuple[int, int]]:
    if n < 2:
        raise ValueError(f"need n>=2 nodes, got {n}")
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.6]
    return edges or [(0, 1)]


def maxcut_cost(x: int, edges: list[tuple[int, int]]) -> float:
    if x < 0:
        raise ValueError(f"bitstring state must be >=0, got {x}")
    return float(sum(((x >> i) & 1) != ((x >> j) & 1) for i, j in edges))
