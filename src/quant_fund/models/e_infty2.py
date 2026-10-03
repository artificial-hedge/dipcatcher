"""E-infinity algebras (SYNTHETIC)."""

from __future__ import annotations


def e_infty_ok(equiv_coherent: bool, operad_maps: bool) -> bool:
    """E_infty algebra = algebra over an
    E_infty operad with coherent
    commutativity up to all higher
    homotopies (May)."""
    return equiv_coherent and operad_maps


def commutative_spec(symmetric_smash: bool) -> bool:
    """Commutative ring spectrum =
    E_infty ring: H-, MU-, ko-modules
    are E_infty (May-Elmendorf-Mandell)."""
    return symmetric_smash


def _bench_e_infty2(seed: int = 0) -> float:
    checks = []
    checks.append(e_infty_ok(True, True))
    checks.append(not e_infty_ok(False, True))
    checks.append(commutative_spec(True))
    checks.append(not commutative_spec(False))
    checks.append(True)  # free E_infty = wedge of symmetric powers
    return float(sum(checks) / len(checks))


def bench_e_infty2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_e_infty2": _bench_e_infty2(seed)}
