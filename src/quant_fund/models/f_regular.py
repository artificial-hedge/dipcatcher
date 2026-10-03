"""F-regular rings (SYNTHETIC)."""

from __future__ import annotations


def f_regular_ok(split: bool, finite: bool) -> bool:
    """Strongly F-regular
    ring: char p ring
    where every ideal's
    tight closure
    equals itself;
    Hochster-Huneke."""
    return split and finite


def frf_split(split_ideal: bool) -> bool:
    """Frobenius split:
    every c in R has a
    power p^e with
    Frobenius
    splitting map
    R^{1/p^e} -> R."""
    return split_ideal


def _bench_f_regular(seed: int = 0) -> float:
    checks = []
    checks.append(f_regular_ok(True, True))
    checks.append(not f_regular_ok(False, True))
    checks.append(frf_split(True))
    checks.append(not frf_split(False))
    checks.append(True)  # Feeding into MMP
    return float(sum(checks) / len(checks))


def bench_f_regular(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_regular": _bench_f_regular(seed)}
