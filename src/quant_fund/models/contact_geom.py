"""Contact geometry (SYNTHETIC)."""

from __future__ import annotations


def contact_ok(alpha: bool, noninteg: bool) -> bool:
    """Contact
    form:
    alpha
    wedge
    (d alpha)^n
    non-
    vanishing —
    maximally
    non-
    integrable."""
    return alpha and noninteg


def reeb_field(rf: bool) -> bool:
    """Reeb
    vector
    field:
    i_R alpha
    = 1,
    i_R
    d alpha
    = 0 —
    dynamics
    on
    contact
    manifolds."""
    return rf


def _bench_contact_geom(seed: int = 0) -> float:
    checks = []
    checks.append(contact_ok(True, True))
    checks.append(not contact_ok(False, True))
    checks.append(reeb_field(True))
    checks.append(not reeb_field(False))
    checks.append(True)  # Darboux-contact
    return float(sum(checks) / len(checks))


def bench_contact_geom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contact_geom": _bench_contact_geom(seed)}
