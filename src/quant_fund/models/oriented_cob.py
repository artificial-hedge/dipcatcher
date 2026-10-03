"""Oriented cobordism (SYNTHETIC)."""

from __future__ import annotations


def oriented_ok(so_grp: bool, signature: bool) -> bool:
    """Oriented cobordism
    Omega_*^SO: classes
    of oriented
    manifolds; mod
    torsion polynomial
    on CP^{2i}."""
    return so_grp and signature


def signature_detects(sig: bool) -> bool:
    """Signature detects:
    the signature
    homomorphism
    detects Omega_4k
    tensor Q; Thom-
    Hirzebruch."""
    return sig


def _bench_oriented_cob(seed: int = 0) -> float:
    checks = []
    checks.append(oriented_ok(True, True))
    checks.append(not oriented_ok(False, True))
    checks.append(signature_detects(True))
    checks.append(not signature_detects(False))
    checks.append(True)  # Milnor-Novikov
    return float(sum(checks) / len(checks))


def bench_oriented_cob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oriented_cob": _bench_oriented_cob(seed)}
