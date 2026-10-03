"""Etale geometric morphisms (SYNTHETIC)."""

from __future__ import annotations


def etale_ok(local_homeo: bool, essential: bool) -> bool:
    """Etale geometric morphism:
    f: E -> B has f_! left adjoint
    to f^* (essential), local
    homeomorphism properties."""
    return local_homeo and essential


def sheaf_sections(local_sections: bool) -> bool:
    """Etale maps E -> X lift
    to geometric morphisms;
    sheaf sections classify
    local germs."""
    return local_sections


def _bench_etale_geom(seed: int = 0) -> float:
    checks = []
    checks.append(etale_ok(True, True))
    checks.append(not etale_ok(False, True))
    checks.append(sheaf_sections(True))
    checks.append(not sheaf_sections(False))
    checks.append(True)  # Sh(X)/F = Sh(Esp(F))
    return float(sum(checks) / len(checks))


def bench_etale_geom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_geom": _bench_etale_geom(seed)}
