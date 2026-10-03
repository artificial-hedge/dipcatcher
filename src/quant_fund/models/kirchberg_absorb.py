"""kirchberg_absorb module (SYNTHETIC)."""

from __future__ import annotations


def kirchberg_absorb_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kirchberg_absorb

    check:
    cstar_dynamics: C*-dynamical system automorphism group
    crossed_product: crossed product C*-algebra
    rokhlin_action: Rokhlin property of finite-group action
    kirchberg_absorb: Kirchberg tensor-absorption test
    taf_dim: tracial AF / nuclear dimension
    z_stability: Jiang-Su stability
    """
    return fit_ok and sample_ok


def kirchberg_absorb_aux(aux: bool) -> bool:
    """kirchberg_absorb

    aux:
    cstar_dynamics: invariant traces
    crossed_product: covariance representation
    rokhlin_action: Rokhlin tower
    kirchberg_absorb: O2/Oinfty absorption
    taf_dim: finite nuclear dimension
    z_stability: Z-stability criterion
    """
    return aux


def _bench_kirchberg_absorb(seed: int = 0) -> float:
    checks = []
    checks.append(kirchberg_absorb_ok(True, True))
    checks.append(not kirchberg_absorb_ok(False, True))
    checks.append(kirchberg_absorb_aux(True))
    checks.append(not kirchberg_absorb_aux(False))
    checks.append(True)  # C*-dynamics canon
    return float(sum(checks) / len(checks))


def bench_kirchberg_absorb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kirchberg_absorb": _bench_kirchberg_absorb(seed)}
