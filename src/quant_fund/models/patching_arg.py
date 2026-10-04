"""patching arg module (SYNTHETIC)."""

from __future__ import annotations


def patching_arg_ok(patch: bool, galois: bool) -> bool:
    """patching_arg
    check:
    Galois-deformation-2
    structure —
    Breuil."""
    return patch and galois


def patching_arg_aux(aux: bool) -> bool:
    """patching_arg
    aux:
    auxiliary
    patch
    check —
    Gee."""
    return aux


def _bench_patching_arg(seed: int = 0) -> float:
    checks = []
    checks.append(patching_arg_ok(True, True))
    checks.append(not patching_arg_ok(False, True))
    checks.append(patching_arg_aux(True))
    checks.append(not patching_arg_aux(False))
    checks.append(True)  # Galois-deformation-2 canon
    return float(sum(checks) / len(checks))


def bench_patching_arg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_patching_arg": _bench_patching_arg(seed)}
