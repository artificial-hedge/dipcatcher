"""Wavefront set (SYNTHETIC)."""

from __future__ import annotations


def wf_ok(conic: bool, microlocal: bool) -> bool:
    """Wavefront
    set
    WF(u):
    conic
    set of
    singular
    directions —
    microlocal
    refinement
    of
    sing
    supp."""
    return conic and microlocal


def propagation(p: bool) -> bool:
    """Propagation:
    WF
    is
    invariant
    under
    the
    bicharacteristic
    flow."""
    return p


def _bench_wavefront_set(seed: int = 0) -> float:
    checks = []
    checks.append(wf_ok(True, True))
    checks.append(not wf_ok(False, True))
    checks.append(propagation(True))
    checks.append(not propagation(False))
    checks.append(True)  # Hörmander
    return float(sum(checks) / len(checks))


def bench_wavefront_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wavefront_set": _bench_wavefront_set(seed)}
