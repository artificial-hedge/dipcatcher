"""Phase-plane classification of 2x2 linear systems by eigenvalues (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def classify(a: np.ndarray) -> str:
    """stable_node | unstable_node | saddle | stable_spiral | unstable_spiral | center."""
    w = np.linalg.eigvals(a)
    re = [float(v.real) for v in w]
    im = [float(v.imag) for v in w]
    if abs(im[0]) > 1e-9:
        if re[0] < -1e-9:
            return "stable_spiral"
        if re[0] > 1e-9:
            return "unstable_spiral"
        return "center"
    if re[0] * re[1] < 0:
        return "saddle"
    if max(re) < 0:
        return "stable_node"
    if min(re) > 0:
        return "unstable_node"
    return "degenerate"


def trace_det_classify(a: np.ndarray) -> str:
    tr = float(np.trace(a))
    det = float(np.linalg.det(a))
    disc = tr * tr - 4 * det
    if det < 0:
        return "saddle"
    if disc < 0:
        if tr < 0:
            return "stable_spiral"
        if tr > 0:
            return "unstable_spiral"
        return "center"
    if tr < 0:
        return "stable_node"
    return "unstable_node"


def _bench_phase_plane(seed: int = 0) -> float:
    checks = []
    checks.append(classify(np.array([[-1.0, 0.0], [0.0, -2.0]])) == "stable_node")
    checks.append(classify(np.array([[1.0, 0.0], [0.0, 2.0]])) == "unstable_node")
    checks.append(classify(np.array([[1.0, 0.0], [0.0, -1.0]])) == "saddle")
    checks.append(classify(np.array([[-1.0, -2.0], [2.0, -1.0]])) == "stable_spiral")
    checks.append(classify(np.array([[0.0, -1.0], [1.0, 0.0]])) == "center")
    # two methods agree on all samples
    for _ in range(50):
        a = np.random.default_rng(_).standard_normal((2, 2))
        checks.append(classify(a) == trace_det_classify(a))
    return float(sum(checks) / len(checks))


def bench_phase_plane(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phase_plane": _bench_phase_plane(seed)}
