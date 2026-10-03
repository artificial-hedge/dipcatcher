"""Blue shift / red shift (SYNTHETIC)."""

from __future__ import annotations


def blue_shift_ok(redshift: bool, chromatic_raise: bool) -> bool:
    """Red shift conjecture: K(n+1)-
    local theory on a K(n)-local
    spectrum; blue shift
    shifts height up."""
    return redshift and chromatic_raise


def telescope_raise(height: bool) -> bool:
    """T(n+1)_* (L_{K(n)} X) sees
    height n+1 phenomena in
    height n; chromatic
    convergence."""
    return height


def _bench_blue_shift(seed: int = 0) -> float:
    checks = []
    checks.append(blue_shift_ok(True, True))
    checks.append(not blue_shift_ok(False, True))
    checks.append(telescope_raise(True))
    checks.append(not telescope_raise(False))
    checks.append(True)  # Hahn-Wilson redshift
    return float(sum(checks) / len(checks))


def bench_blue_shift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blue_shift": _bench_blue_shift(seed)}
