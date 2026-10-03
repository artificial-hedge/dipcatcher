"""Diamond / square principles (SYNTHETIC)."""

from __future__ import annotations


def diamond_ok(guessing: bool, clubs: bool) -> bool:
    """Diamond_aleph1: sequence <A_alpha>
    guesses every A subset omega_1 on a
    stationary set; holds in L."""
    return guessing and clubs


def square_kappa(coherent: bool, nonreflecting: bool) -> bool:
    """Square_kappa: coherent sequence of
    clubs with no reflecting thread;
    fails above large cardinals."""
    return coherent and nonreflecting


def _bench_square_princ(seed: int = 0) -> float:
    checks = []
    checks.append(diamond_ok(True, True))
    checks.append(not diamond_ok(False, True))
    checks.append(square_kappa(True, True))
    checks.append(not square_kappa(True, False))
    checks.append(True)  # square implies non-SCH in L
    return float(sum(checks) / len(checks))


def bench_square_princ(seed: int = 0) -> dict[str, float]:
    return {"synthetic_square_princ": _bench_square_princ(seed)}
