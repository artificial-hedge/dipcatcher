"""Two-qubit entanglement: concurrence, negativity, separability (SYNTHETIC bench)."""

from __future__ import annotations

import numpy as np


def concurrence(rho: np.ndarray) -> float:
    sy = np.array([[0, -1j], [1j, 0]], complex)
    Y = np.kron(sy, sy)
    rt = rho @ Y @ rho.conj() @ Y
    ev = np.sqrt(np.clip(np.linalg.eigvalsh(rt), 0, None))
    srt = sorted(ev.tolist(), reverse=True)
    return float(max(0.0, srt[0] - srt[1] - srt[2] - srt[3]))


def partial_transpose(rho: np.ndarray, sys: int = 1) -> np.ndarray:
    t = rho.reshape(2, 2, 2, 2)
    if sys == 1:
        t = t.transpose(0, 3, 2, 1)
    else:
        t = t.transpose(2, 1, 0, 3)
    return t.reshape(4, 4)


def negativity(rho: np.ndarray) -> float:
    pt = partial_transpose(rho, 1)
    ev = np.linalg.eigvalsh(pt)
    return float(np.sum(np.abs(ev[ev < 0])))


def is_separable(rho: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(negativity(rho) < tol)


def _bench_entanglement(seed: int = 0) -> float:
    checks = []
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rb = np.outer(bell, bell.conj())
    checks.append(abs(concurrence(rb) - 1.0) < 1e-9)
    checks.append(abs(negativity(rb) - 0.5) < 1e-9)
    prod = np.kron(np.array([[1, 0], [0, 0]], complex), np.array([[1, 0], [0, 0]], complex))
    checks.append(abs(concurrence(prod)) < 1e-9)
    checks.append(is_separable(prod))
    # maximally mixed -> separable
    checks.append(is_separable(np.eye(4) / 4))
    # Werner state rho = p|bell><bell| + (1-p)I/4: entangled iff p > 1/3
    for p, ent in [(0.9, True), (0.2, False)]:
        rw = p * rb + (1 - p) * np.eye(4) / 4
        checks.append((not is_separable(rw)) == ent)
    return sum(checks) / len(checks)


def bench_entanglement(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entanglement": _bench_entanglement(seed)}
