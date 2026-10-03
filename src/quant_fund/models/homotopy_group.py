"""pi_n(S^n) = Z via degree; composition adds degrees (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def degree_s1(pts: np.ndarray) -> int:
    """Degree of a loop gamma: S^1 -> S^1 given unit samples."""
    th = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
    return int(round((th[-1] - th[0]) / (2 * np.pi)))


def _bench_homotopy_group(seed: int = 0) -> float:
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    checks = []
    # gamma_k(t) = e^{ikt}: degree k
    for k in (-2, -1, 0, 1, 3):
        pts = np.stack([np.cos(k * t), np.sin(k * t)], axis=1)
        checks.append(degree_s1(pts) == k)
    # concatenation adds: gamma_2 . gamma_3 ~ degree 5
    p2 = np.stack([np.cos(2 * t), np.sin(2 * t)], axis=1)
    p3 = np.stack([np.cos(3 * t), np.sin(3 * t)], axis=1)
    z2 = p2[:, 0] + 1j * p2[:, 1]
    z3 = p3[:, 0] + 1j * p3[:, 1]
    # complex multiplication on S^1 = group operation on pi_1
    prod = np.stack([(z2 * z3).real, (z2 * z3).imag], axis=1)
    checks.append(degree_s1(prod) == 5)
    # inverse = conjugate: gamma_k * conj(gamma_k) = constant -> degree 0
    inv = np.stack([(z2 * np.conj(z2)).real, (z2 * np.conj(z2)).imag], axis=1)
    checks.append(degree_s1(inv) == 0)
    # small perturbation preserves degree (homotopy invariance)
    rng = np.random.default_rng(seed)
    p2n = p2 + 0.05 * rng.normal(size=p2.shape)
    p2n /= np.linalg.norm(p2n, axis=1, keepdims=True)
    checks.append(degree_s1(p2n) == 2)
    return float(sum(checks) / len(checks))


def bench_homotopy_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_group": _bench_homotopy_group(seed)}
