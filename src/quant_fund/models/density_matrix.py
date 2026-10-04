"""Density-matrix mechanics: purity, partial trace, fidelity (SYNTHETIC bench)."""

from __future__ import annotations

import math

import numpy as np


def pure_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conj(psi))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    """Trace out all subsystems except `keep` (dims per subsystem)."""
    n = len(dims)
    r = rho.reshape(dims + dims)
    trace_out = sorted(set(range(n)) - set(keep), reverse=True)
    for ax in trace_out:
        r = np.trace(r, axis1=ax, axis2=ax + len(r.shape) // 2)
    dk = int(np.prod([dims[i] for i in keep]))
    return r.reshape(dk, dk)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity (pure-friendly): Tr sqrt(sqrt(rho) sigma sqrt(rho))^2."""
    if purity(rho) > 0.999:
        v = rho
        w = np.linalg.eigvalsh(v)
        idx = int(np.argmax(w))
        s = np.linalg.eigh(v)[1][:, idx]
        return float(np.real(np.conj(s) @ sigma @ s))
    ev, U = np.linalg.eigh(rho)
    sq = U @ np.diag(np.sqrt(np.clip(ev, 0, None))) @ U.conj().T
    mid = sq @ sigma @ sq
    ev2 = np.linalg.eigvalsh(mid)
    return float(np.real(np.sum(np.sqrt(np.clip(ev2, 0, None)))) ** 2)


def is_valid(rho: np.ndarray, tol: float = 1e-9) -> bool:
    return (
        np.allclose(rho, rho.conj().T, atol=tol)
        and abs(float(np.real(np.trace(rho))) - 1.0) < tol
        and float(np.min(np.linalg.eigvalsh(rho))) > -tol
    )


def _bench_density_matrix(seed: int = 0) -> float:
    checks = []
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    rho_b = pure_state(bell)
    checks.append(is_valid(rho_b))
    checks.append(abs(purity(rho_b) - 1.0) < 1e-9)
    # partial trace of a Bell state -> maximally mixed
    red = partial_trace(rho_b, [0], [2, 2])
    checks.append(np.allclose(red, np.eye(2) / 2))
    checks.append(abs(purity(red) - 0.5) < 1e-9)
    # fidelity: identical pure states -> 1; orthogonal -> 0
    checks.append(abs(fidelity(rho_b, rho_b) - 1.0) < 1e-9)
    rho0 = pure_state(np.array([1, 0], dtype=complex))
    rho1 = pure_state(np.array([0, 1], dtype=complex))
    checks.append(abs(fidelity(rho0, rho1)) < 1e-9)
    checks.append(not is_valid(np.array([[1.1, 0], [0, 0]], dtype=complex)))
    return sum(checks) / len(checks)


def bench_density_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_density_matrix": _bench_density_matrix(seed)}
