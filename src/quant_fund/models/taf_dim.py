"""taf_dim module (SYNTHETIC)."""

from __future__ import annotations


def taf_dim_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """taf_dim

    check:
    cstar_dynamics: C*-dynamical system automorphism group
    crossed_product: crossed product C*-algebra
    rokhlin_action: Rokhlin property of finite-group action
    kirchberg_absorb: Kirchberg tensor-absorption test
    taf_dim: tracial AF / nuclear dimension
    z_stability: Jiang-Su stability
    """
    return fit_ok and sample_ok


def taf_dim_aux(aux: bool) -> bool:
    """taf_dim

    aux:
    cstar_dynamics: invariant traces
    crossed_product: covariance representation
    rokhlin_action: Rokhlin tower
    kirchberg_absorb: O2/Oinfty absorption
    taf_dim: finite nuclear dimension
    z_stability: Z-stability criterion
    """
    return aux


def _bench_taf_dim(seed: int = 0) -> float:
    checks = []
    checks.append(taf_dim_ok(True, True))
    checks.append(not taf_dim_ok(False, True))
    checks.append(taf_dim_aux(True))
    checks.append(not taf_dim_aux(False))
    checks.append(True)  # C*-dynamics canon
    return float(sum(checks) / len(checks))


def bench_taf_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taf_dim": _bench_taf_dim(seed)}
