"""Quantum channels via Kraus operators + TP verification (SYNTHETIC bench)."""

from __future__ import annotations

import numpy as np


def apply(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(rho)
    for k in kraus:
        out = out + k @ rho @ k.conj().T
    return out


def is_tp(kraus: list[np.ndarray], tol: float = 1e-9) -> bool:
    s = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(s, np.eye(len(s)), atol=tol))


def depolarizing(p: float) -> list[np.ndarray]:
    """Single-qubit depolarizing channel Kraus ops."""
    k0 = np.eye(2)
    kx = np.array([[0, 1], [1, 0]], complex)
    ky = np.array([[0, -1j], [1j, 0]], complex)
    kz = np.array([[1, 0], [0, -1]], complex)
    return [
        np.sqrt(1 - 3 * p / 4) * k0,
        np.sqrt(p / 4) * kx,
        np.sqrt(p / 4) * ky,
        np.sqrt(p / 4) * kz,
    ]


def amplitude_damping(g: float) -> list[np.ndarray]:
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], complex)
    K1 = np.array([[0, np.sqrt(g)], [0, 0]], complex)
    return [K0, K1]


def compose(k1: list[np.ndarray], k2: list[np.ndarray]) -> list[np.ndarray]:
    return [a @ b for a in k1 for b in k2]


def _bench_qchannel(seed: int = 0) -> float:
    checks = []
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    checks.append(is_tp(depolarizing(0.2)))
    checks.append(is_tp(amplitude_damping(0.3)))
    out = apply(rho, depolarizing(0.4))
    checks.append(abs(float(np.real(np.trace(out))) - 1.0) < 1e-9)
    checks.append(is_tp([np.eye(2)]))
    checks.append(not is_tp([np.eye(2) * 0.5]))
    # amplitude damping shrinks excited population
    exc = np.array([[0, 0], [0, 1]], dtype=complex)
    out2 = apply(exc, amplitude_damping(0.5))
    checks.append(abs(float(out2[1, 1].real) - 0.5) < 1e-9)
    checks.append(abs(float(out2[0, 0].real) - 0.5) < 1e-9)
    # full depolarizing p=1 sends to maximally mixed
    out3 = apply(rho, depolarizing(1.0))
    checks.append(np.allclose(out3, np.eye(2) / 2, atol=1e-9))
    checks.append(is_tp(compose(amplitude_damping(0.1), depolarizing(0.1))))
    return sum(checks) / len(checks)


def bench_qchannel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qchannel": _bench_qchannel(seed)}
