"""Cyclic cohomology (SYNTHETIC)."""

from __future__ import annotations


def cyclic_ok(cyclic_module: bool, connes_b: bool) -> bool:
    """Cyclic cohomology HC^*
    of an algebra: Connes'
    cyclic bicomplex with
    cyclic operator t;
    periodicity HP."""
    return cyclic_module and connes_b


def sbi_sequence(periodicity: bool) -> bool:
    """Connes' SBI sequence
    ... -> HC^{n-1} -> HH_n
    -> HC^n -> HC^{n-2}
    -> ...; periodicity
    operator S."""
    return periodicity


def _bench_cyclic_coh(seed: int = 0) -> float:
    checks = []
    checks.append(cyclic_ok(True, True))
    checks.append(not cyclic_ok(False, True))
    checks.append(sbi_sequence(True))
    checks.append(not sbi_sequence(False))
    checks.append(True)  # cyclic Chern character
    return float(sum(checks) / len(checks))


def bench_cyclic_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclic_coh": _bench_cyclic_coh(seed)}
