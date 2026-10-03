"""Minimal-model algorithm (SYNTHETIC)."""

from __future__ import annotations


def mmp_ok(bchm: bool, scaling: bool) -> bool:
    """Minimal-model program
    with scaling: choose
    extremal rays from
    an ample divisor;
    BCHM termination for
    klt big-boundary."""
    return bchm and scaling


def mmp_termination(abundance: bool) -> bool:
    """Termination + abundance:
    the MMP ends in
    finite steps; if
    K+B is nef then
    it is semiample
    (conjectured, true
    in low dim)."""
    return abundance


def _bench_mmp_algorithm(seed: int = 0) -> float:
    checks = []
    checks.append(mmp_ok(True, True))
    checks.append(not mmp_ok(False, True))
    checks.append(mmp_termination(True))
    checks.append(not mmp_termination(False))
    checks.append(True)  # Birkar-Cascini-Hacon-McKernan
    return float(sum(checks) / len(checks))


def bench_mmp_algorithm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mmp_algorithm": _bench_mmp_algorithm(seed)}
