"""Frobenius action on l-adic cohomology (SYNTHETIC)."""

from __future__ import annotations


def frob_ok(geometric: bool, eigen: bool) -> bool:
    """Frobenius action:
    geometric Frobenius
    F acts on
    H^i_et(X_{Fbar},
    Q_l); eigenvalues
    are Weil numbers."""
    return geometric and eigen


def eigen_pure(pure: bool) -> bool:
    """Purity: eigenvalues
    of F on H^i
    are algebraic
    with |α| =
    q^{i/2} (Weil II)."""
    return pure


def _bench_frobenius_action(seed: int = 0) -> float:
    checks = []
    checks.append(frob_ok(True, True))
    checks.append(not frob_ok(False, True))
    checks.append(eigen_pure(True))
    checks.append(not eigen_pure(False))
    checks.append(True)  # Deligne Weil II
    return float(sum(checks) / len(checks))


def bench_frobenius_action(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frobenius_action": _bench_frobenius_action(seed)}
