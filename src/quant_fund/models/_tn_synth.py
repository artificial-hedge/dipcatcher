"""Shared fixture for wave-200 tensor-network canon — smooth multivariate
tensors + TFIM Hamiltonian + MPS helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def smooth_tensor(seed: int, d: int = 4, n: int = 8) -> FloatArray:
    """Low-TT-rank smooth tensor: sum of 3 separable Gaussian bumps."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    T = np.zeros([n] * d)
    for _ in range(3):
        c = rng.uniform(0.2, 0.8, d)
        a = rng.uniform(0.5, 1.5)
        bump = np.ones([n] * d)
        for k in range(d):
            shape = [1] * d
            shape[k] = n
            bump = bump * np.exp(-a * (x.reshape(shape) - c[k]) ** 2)
        T += bump
    return np.asarray(T)


def tfim_h(n: int, h: float = 1.0) -> FloatArray:
    """Transverse-field Ising Hamiltonian d=2^n."""
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    ID = np.eye(2)
    ham = np.zeros((2**n, 2**n))
    for q in range(n - 1):
        ops = [ID] * n
        ops[q], ops[q + 1] = Z, Z
        op: np.ndarray = ops[0]
        for o in ops[1:]:
            op = np.kron(op, o)
        ham -= op
    for q in range(n):
        ops = [ID] * n
        ops[q] = X
        op = ops[0]
        for o in ops[1:]:
            op = np.kron(op, o)
        ham -= h * op
    return np.asarray(ham)


def to_mps(T: np.ndarray, d_phys: int = 2, chi: int = 8) -> list[np.ndarray]:
    """Flatten a d^n tensor to an MPS via sequential SVD (left-canonical)."""
    n = int(round(np.log(T.size) / np.log(d_phys)))
    psi = T.reshape(d_phys, -1)
    cores = []
    W = psi.reshape(1, d_phys, -1)
    rem = W
    for _ in range(n - 1):
        s = rem.shape
        M = rem.reshape(s[0] * d_phys, -1)
        U, S, V = np.linalg.svd(M, full_matrices=False)
        r = min(chi, len(S))
        U, S, V = U[:, :r], S[:r], V[:r]
        cores.append(U.reshape(s[0], d_phys, r))
        rem = (np.diag(S) @ V).reshape(r, d_phys, -1)
    cores.append(rem.reshape(rem.shape[0], d_phys, -1))
    return cores


def mps_contract(cores: list[np.ndarray]) -> np.ndarray:
    out = cores[0]
    for c in cores[1:]:
        out = np.tensordot(out, c, axes=(-1, 0))
    return np.asarray(out.reshape(-1))
