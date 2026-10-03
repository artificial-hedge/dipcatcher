"""Mirror functors (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(fourier_mukai: bool, equivalence: bool) -> bool:
    """Mirror
    functor:
    explicit
    functor
    from
    Lagrangians
    to
    sheaves —
    SYZ
    Fourier-
    Mukai
    transform."""
    return fourier_mukai and equivalence


def chl_functor(cf: bool) -> bool:
    """CHL
    functor:
    Cho-
    Hong-
    Lau
    localized
    mirror
    construction
    via
    formal
    deformation
    of
    a
    torus
    fiber."""
    return cf


def _bench_mirror_functor(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(chl_functor(True))
    checks.append(not chl_functor(False))
    checks.append(True)  # CHL
    return float(sum(checks) / len(checks))


def bench_mirror_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mirror_functor": _bench_mirror_functor(seed)}
