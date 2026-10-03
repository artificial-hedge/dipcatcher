"""Azure spectral spaces (SYNTHETIC)."""

from __future__ import annotations


def az_ok(azure: bool, space: bool) -> bool:
    """Azure
    space:
    azure
    spectral
    space —
    Lurie."""
    return azure and space


def lurie_spectral(ls: bool) -> bool:
    """Lurie
    spectral:
    Lurie
    spectral
    space —
    spectral
    scheme."""
    return ls


def _bench_azure_space(seed: int = 0) -> float:
    checks = []
    checks.append(az_ok(True, True))
    checks.append(not az_ok(False, True))
    checks.append(lurie_spectral(True))
    checks.append(not lurie_spectral(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_azure_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_azure_space": _bench_azure_space(seed)}
