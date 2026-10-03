"""POVM measurement: element sets, completeness, outcome probabilities (SYNTHETIC bench)."""

from __future__ import annotations

import numpy as np


def is_povm(elems: list[np.ndarray], tol: float = 1e-9) -> bool:
    s = np.zeros_like(elems[0])
    for e in elems:
        s = s + e
    return bool(
        np.allclose(s, np.eye(s.shape[0]), atol=tol)
        and all(float(np.min(np.linalg.eigvalsh(e))) > -tol for e in elems)
    )


def probs(rho: np.ndarray, elems: list[np.ndarray]) -> list[float]:
    return [float(np.real(np.trace(e @ rho))) for e in elems]


def povm_from_effects(effs: list[np.ndarray]) -> list[np.ndarray]:
    return effs


def projective(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, np.conj(ket))


def _bench_povm_measure(seed: int = 0) -> float:
    checks = []
    # computational-basis POVM
    z = [projective(np.array([1, 0], complex)), projective(np.array([0, 1], complex))]
    checks.append(is_povm(z))
    rho = np.array([[0.75, 0], [0, 0.25]], dtype=complex)
    checks.append(np.allclose(probs(rho, z), [0.75, 0.25]))
    # three-outcome qubit POVM (non-projective): E_i = (2/3)|v_i><v_i|, v_i 120deg apart
    vs = [
        np.array([1.0, 0.0], complex),
        np.array([-0.5, np.sqrt(3) / 2], complex),
        np.array([-0.5, -np.sqrt(3) / 2], complex),
    ]
    trine = [(2 / 3) * projective(v) for v in vs]
    checks.append(is_povm(trine))
    p = probs(projective(np.array([1, 0], complex)), trine)
    checks.append(abs(sum(p) - 1.0) < 1e-9 and abs(p[0] - 2 / 3) < 1e-9)
    # completeness failure detection
    checks.append(not is_povm([z[0]]))
    # negative eigenvalue rejection
    checks.append(not is_povm([np.eye(2), np.array([[0, 0], [0, -0.1]], complex)]))
    return sum(checks) / len(checks)


def bench_povm_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_povm_measure": _bench_povm_measure(seed)}
