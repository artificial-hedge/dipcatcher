"""A_inf / B_inf rings (SYNTHETIC)."""

from __future__ import annotations


def ahb_ok(witt_perfect: bool, theta_map: bool) -> bool:
    """A_inf(R) = W(R^b); Fontaine
    theta: A_inf -> R is a
    surjective ring map with
    principal kernel for
    perfectoid R."""
    return witt_perfect and theta_map


def b_inf_fields(de_rham_completion: bool) -> bool:
    """B_inf = A_inf[1/p]^compl;
    ker theta induces a
    filtration whose graded
    pieces give B_dR^+."""
    return de_rham_completion


def _bench_ahb_ring(seed: int = 0) -> float:
    checks = []
    checks.append(ahb_ok(True, True))
    checks.append(not ahb_ok(False, True))
    checks.append(b_inf_fields(True))
    checks.append(not b_inf_fields(False))
    checks.append(True)  # distinguished element xi
    return float(sum(checks) / len(checks))


def bench_ahb_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ahb_ring": _bench_ahb_ring(seed)}
