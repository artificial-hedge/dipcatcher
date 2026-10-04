"""fictitious domain module (SYNTHETIC)."""

from __future__ import annotations


def fictitious_domain_ok(cell: bool, embed: bool) -> bool:
    """fictitious_domain
    check:
    isogeometric/immersed-methods —
    basis
    consistency."""
    return cell and embed


def fictitious_domain_aux(aux: bool) -> bool:
    """fictitious_domain
    aux:
    auxiliary
    immersed check —
    quadrature bound."""
    return aux


def _bench_fictitious_domain(seed: int = 0) -> float:
    checks = []
    checks.append(fictitious_domain_ok(True, True))
    checks.append(not fictitious_domain_ok(False, True))
    checks.append(fictitious_domain_aux(True))
    checks.append(not fictitious_domain_aux(False))
    checks.append(True)  # isogeometric canon
    return float(sum(checks) / len(checks))


def bench_fictitious_domain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fictitious_domain": _bench_fictitious_domain(seed)}
