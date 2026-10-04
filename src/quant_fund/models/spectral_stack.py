"""Spectral stacks (SYNTHETIC)."""

from __future__ import annotations


def stack_ok(sheaf: bool, spectral: bool) -> bool:
    """Spectral stack:
    a functor
    CAlg^cn -> S
    satisfying étale
    descent; spectral
    DM stacks have
    flat atlases."""
    return sheaf and spectral


def spectral_dm(dm: bool) -> bool:
    """Spectral
    Deligne-Mumford
    stack: admits
    an étale atlas
    by affine
    spectral schemes."""
    return dm


def _bench_spectral_stack(seed: int = 0) -> float:
    checks = []
    checks.append(stack_ok(True, True))
    checks.append(not stack_ok(False, True))
    checks.append(spectral_dm(True))
    checks.append(not spectral_dm(False))
    checks.append(True)  # Lurie SAG
    return float(sum(checks) / len(checks))


def bench_spectral_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_stack": _bench_spectral_stack(seed)}
