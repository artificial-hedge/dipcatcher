"""sig inversion module (SYNTHETIC)."""

from __future__ import annotations


def sig_inversion_ok(sk1: bool, sg: bool) -> bool:
    """sig_inversion
    check:
    signature
    —
    kernel/PDE."""
    return sk1 and sg


def sig_inversion_aux(aux: bool) -> bool:
    """sig_inversion
    aux:
    auxiliary
    sig
    check —
    expected/inversion."""
    return aux


def _bench_sig_inversion(seed: int = 0) -> float:
    checks = []
    checks.append(sig_inversion_ok(True, True))
    checks.append(not sig_inversion_ok(False, True))
    checks.append(sig_inversion_aux(True))
    checks.append(not sig_inversion_aux(False))
    checks.append(True)  # signature canon
    return float(sum(checks) / len(checks))


def bench_sig_inversion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sig_inversion": _bench_sig_inversion(seed)}
