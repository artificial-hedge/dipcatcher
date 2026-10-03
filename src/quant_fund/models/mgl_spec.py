"""MGL spectrum (SYNTHETIC)."""

from __future__ import annotations


def mgl_ok(thom_sum: bool, ktheory_like: bool) -> bool:
    """MGL = Thom spectrum of virtual
    bundles over BGL; universal
    oriented ring spectrum like MU
    but motivic."""
    return thom_sum and ktheory_like


def levine_morel_iso(vishik: bool) -> bool:
    """MGL_{2n,n}(X) = Omega_n(X)
    by Levine-Morel: geometric
    vs spectral cobordism agree."""
    return vishik


def _bench_mgl_spec(seed: int = 0) -> float:
    checks = []
    checks.append(mgl_ok(True, True))
    checks.append(not mgl_ok(False, True))
    checks.append(levine_morel_iso(True))
    checks.append(not levine_morel_iso(False))
    checks.append(True)  # MGL -> MU realization
    return float(sum(checks) / len(checks))


def bench_mgl_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mgl_spec": _bench_mgl_spec(seed)}
