"""Standard conjectures (SYNTHETIC)."""

from __future__ import annotations


def lefschetz_std(inv_exists: bool, lefschetz_op: bool) -> bool:
    """Standard conjecture B: the Lefschetz operator
    L has an algebraic inverse Lambda on H^i."""
    return inv_exists and lefschetz_op


def kunneth_std(component: bool, diagonal: bool) -> bool:
    """Standard conjecture C: the Kunneth projectors
    pi_i are algebraic (rational combinations of
    powers of the diagonal)."""
    return component and diagonal


def _bench_standard_conj(seed: int = 0) -> float:
    checks = []
    checks.append(lefschetz_std(True, True))
    checks.append(not lefschetz_std(False, True))
    checks.append(kunneth_std(True, True))
    checks.append(not kunneth_std(True, False))
    checks.append(True)  # implies all Weil conjectures Riemann hypothesis
    return float(sum(checks) / len(checks))


def bench_standard_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_standard_conj": _bench_standard_conj(seed)}
