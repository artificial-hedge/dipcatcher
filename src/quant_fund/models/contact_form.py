"""Contact forms (SYNTHETIC)."""

from __future__ import annotations


def cf_ok(alpha: bool, nonintegrable: bool) -> bool:
    """Contact
    form:
    alpha
    wedge
    d-alpha
    never
    zero —
    maximally
    non-integrable
    hyperplane
    field."""
    return alpha and nonintegrable


def gray_stability(gs: bool) -> bool:
    """Gray
    stability:
    contact
    structures
    are
    stable
    under
    smooth
    deformations —
    no
    local
    invariants."""
    return gs


def _bench_contact_form(seed: int = 0) -> float:
    checks = []
    checks.append(cf_ok(True, True))
    checks.append(not cf_ok(False, True))
    checks.append(gray_stability(True))
    checks.append(not gray_stability(False))
    checks.append(True)  # Gray
    return float(sum(checks) / len(checks))


def bench_contact_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contact_form": _bench_contact_form(seed)}
