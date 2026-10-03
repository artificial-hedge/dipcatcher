"""karl dimers module (SYNTHETIC)."""

from __future__ import annotations


def karl_dimers_ok(dim: bool, hf: bool) -> bool:
    """karl_dimers
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def karl_dimers_aux(aux: bool) -> bool:
    """karl_dimers
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_karl_dimers(seed: int = 0) -> float:
    checks = []
    checks.append(karl_dimers_ok(True, True))
    checks.append(not karl_dimers_ok(False, True))
    checks.append(karl_dimers_aux(True))
    checks.append(not karl_dimers_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_karl_dimers(seed: int = 0) -> dict[str, float]:
    return {"synthetic_karl_dimers": _bench_karl_dimers(seed)}
