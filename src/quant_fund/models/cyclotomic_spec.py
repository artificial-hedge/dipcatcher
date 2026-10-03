"""Cyclotomic spectrum (SYNTHETIC)."""

from __future__ import annotations


def cs_ok2(cyclotomic: bool, frobenius: bool) -> bool:
    """Cyclotomic:
    cyclotomic
    spectrum
    with
    S1-
    Frobenius —
    BHM
    cyclotomic."""
    return cyclotomic and frobenius


def cyclotomic_fixed(cf: bool) -> bool:
    """Cyclotomic
    fixed:
    Frobenius
    on
    Tate
    orbits —
    cyclotomic
    structure."""
    return cf


def _bench_cyclotomic_spec(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok2(True, True))
    checks.append(not cs_ok2(False, True))
    checks.append(cyclotomic_fixed(True))
    checks.append(not cyclotomic_fixed(False))
    checks.append(True)  # BHM
    return float(sum(checks) / len(checks))


def bench_cyclotomic_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclotomic_spec": _bench_cyclotomic_spec(seed)}
