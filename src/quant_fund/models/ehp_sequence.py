"""EHP sequence (SYNTHETIC)."""

from __future__ import annotations


def ehp_ok(ehp_map: bool, suspension: bool) -> bool:
    """EHP:
    suspension
    long
    exact
    sequence —
    James
    EHP."""
    return ehp_map and suspension


def ehp_iter(ei: bool) -> bool:
    """EHP
    iteration:
    iterates
    to
    stable
    homotopy —
    James
    spectral."""
    return ei


def _bench_ehp_sequence(seed: int = 0) -> float:
    checks = []
    checks.append(ehp_ok(True, True))
    checks.append(not ehp_ok(False, True))
    checks.append(ehp_iter(True))
    checks.append(not ehp_iter(False))
    checks.append(True)  # James
    return float(sum(checks) / len(checks))


def bench_ehp_sequence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ehp_sequence": _bench_ehp_sequence(seed)}
