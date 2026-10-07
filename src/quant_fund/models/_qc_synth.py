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
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    return np.asarray(psi)


def apply1(psi: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    ops = [I2] * n
    ops[qubit] = gate
    return np.asarray(kron_all(*ops) @ psi)


def measure_probs(psi: np.ndarray) -> FloatArray:
    return np.asarray(np.abs(psi) ** 2)


def maxcut_graph(seed: int, n: int = 4) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.6]
    return edges or [(0, 1)]


def maxcut_cost(x: int, edges: list[tuple[int, int]]) -> float:
    return float(sum(((x >> i) & 1) != ((x >> j) & 1) for i, j in edges))
