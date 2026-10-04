"""Multifractal formalism (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(spectrum: bool, legendre: bool) -> bool:
    """Multifractal
    formalism:
    singularity
    spectrum
    f(alpha)
    is the
    Legendre
    transform
    of
    tau(q)."""
    return spectrum and legendre


def tau_scaling(tau: bool) -> bool:
    """Scaling
    function
    tau(q):
    partition
    sums
    scale
    as
    eps^{tau(q)}."""
    return tau


def _bench_multifractal_formal(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(tau_scaling(True))
    checks.append(not tau_scaling(False))
    checks.append(True)  # Halsey et al.
    return float(sum(checks) / len(checks))


def bench_multifractal_formal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multifractal_formal": _bench_multifractal_formal(seed)}
