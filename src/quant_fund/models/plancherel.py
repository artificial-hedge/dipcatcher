"""Plancherel/Parseval identities for the finite DFT (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    k = np.arange(n)
    return np.asarray(np.exp(-2j * np.pi * np.outer(k, k) / n) @ x)


def idft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    k = np.arange(n)
    return np.asarray(np.exp(2j * np.pi * np.outer(k, k) / n) @ x / n)


def parseval_lhs(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x) ** 2))


def parseval_rhs(x: np.ndarray) -> float:
    return float(np.sum(np.abs(dft(x)) ** 2) / len(x))


def _bench_plancherel(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    checks.append(abs(parseval_lhs(x) - parseval_rhs(x)) < 1e-8)
    checks.append(np.allclose(idft(dft(x)), x, atol=1e-10))
    # Parseval for inner products: <x,y> = (1/n)<X,Y>
    y = rng.normal(size=64)
    lhs = float(np.dot(x.real, y))
    rhs = float(np.real(np.vdot(dft(x), dft(y))) / 64)
    checks.append(abs(lhs - rhs) < 1e-8)
    # unit impulse -> flat spectrum
    e = np.zeros(8)
    e[0] = 1.0
    checks.append(np.allclose(np.abs(dft(e)), 1.0))
    # delta_train -> delta_train
    checks.append(np.allclose(np.abs(dft(np.ones(8))), [8, 0, 0, 0, 0, 0, 0, 0], atol=1e-9))
    return float(sum(checks) / len(checks))


def bench_plancherel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plancherel": _bench_plancherel(seed)}
