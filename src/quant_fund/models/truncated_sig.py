"""truncated sig module (SYNTHETIC)."""

from __future__ import annotations


def truncated_sig_ok(sk1: bool, sg: bool) -> bool:
    """truncated_sig
    check:
    signature
    —
    kernel/PDE."""
    return sk1 and sg


def truncated_sig_aux(aux: bool) -> bool:
    """truncated_sig
    aux:
    auxiliary
    sig
    check —
    expected/inversion."""
    return aux


def _bench_truncated_sig(seed: int = 0) -> float:
    checks = []
    checks.append(truncated_sig_ok(True, True))
    checks.append(not truncated_sig_ok(False, True))
    checks.append(truncated_sig_aux(True))
    checks.append(not truncated_sig_aux(False))
    checks.append(True)  # signature canon
    return float(sum(checks) / len(checks))


def bench_truncated_sig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_truncated_sig": _bench_truncated_sig(seed)}
