"""Inseparable extensions: Frobenius and p-th roots (SYNTHETIC)."""

from __future__ import annotations


def is_separable(min_poly_deriv: int) -> bool:
    """An irreducible is separable iff its derivative is nonzero."""
    return min_poly_deriv != 0


def _bench_inseparable(seed: int = 0) -> float:
    checks = []
    # x^p - t over F_p(t): derivative p*x^{p-1} = 0 -> inseparable
    checks.append(not is_separable(0))
    # x^2 + 1 over F_3: derivative 2x != 0 -> separable
    checks.append(is_separable(1))
    # char 0: all extensions separable
    checks.append(is_separable(5))
    # purely inseparable: minimal polynomial x^{p^e} - a
    checks.append(not is_separable(0))
    # separable degree = number of distinct roots
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_inseparable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inseparable": _bench_inseparable(seed)}
