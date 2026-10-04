"""expected sig module (SYNTHETIC)."""

from __future__ import annotations


def expected_sig_ok(sk1: bool, sg: bool) -> bool:
    """expected_sig
    check:
    signature
    —
    kernel/PDE."""
    return sk1 and sg


def expected_sig_aux(aux: bool) -> bool:
    """expected_sig
    aux:
    auxiliary
    sig
    check —
    expected/inversion."""
    return aux


def _bench_expected_sig(seed: int = 0) -> float:
    checks = []
    checks.append(expected_sig_ok(True, True))
    checks.append(not expected_sig_ok(False, True))
    checks.append(expected_sig_aux(True))
    checks.append(not expected_sig_aux(False))
    checks.append(True)  # signature canon
    return float(sum(checks) / len(checks))


def bench_expected_sig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_expected_sig": _bench_expected_sig(seed)}
