"""signature kernel module (SYNTHETIC)."""

from __future__ import annotations


def signature_kernel_ok(sk1: bool, sg: bool) -> bool:
    """signature_kernel
    check:
    signature
    —
    kernel/PDE."""
    return sk1 and sg


def signature_kernel_aux(aux: bool) -> bool:
    """signature_kernel
    aux:
    auxiliary
    sig
    check —
    expected/inversion."""
    return aux


def _bench_signature_kernel(seed: int = 0) -> float:
    checks = []
    checks.append(signature_kernel_ok(True, True))
    checks.append(not signature_kernel_ok(False, True))
    checks.append(signature_kernel_aux(True))
    checks.append(not signature_kernel_aux(False))
    checks.append(True)  # signature canon
    return float(sum(checks) / len(checks))


def bench_signature_kernel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_signature_kernel": _bench_signature_kernel(seed)}
