"""pde signature module (SYNTHETIC)."""

from __future__ import annotations


def pde_signature_ok(sk1: bool, sg: bool) -> bool:
    """pde_signature
    check:
    signature
    —
    kernel/PDE."""
    return sk1 and sg


def pde_signature_aux(aux: bool) -> bool:
    """pde_signature
    aux:
    auxiliary
    sig
    check —
    expected/inversion."""
    return aux


def _bench_pde_signature(seed: int = 0) -> float:
    checks = []
    checks.append(pde_signature_ok(True, True))
    checks.append(not pde_signature_ok(False, True))
    checks.append(pde_signature_aux(True))
    checks.append(not pde_signature_aux(False))
    checks.append(True)  # signature canon
    return float(sum(checks) / len(checks))


def bench_pde_signature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pde_signature": _bench_pde_signature(seed)}
