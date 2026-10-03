"""EK subfactor (SYNTHETIC)."""

from __future__ import annotations


def ek_ok(ek: bool, subfactor: bool) -> bool:
    """EK
    subfactor:
    extended
    Haagerup
    subfactor —
    EK."""
    return ek and subfactor


def extended_haagerup(eh: bool) -> bool:
    """Extended
    Haagerup:
    extended
    Haagerup
    category —
    BMPS."""
    return eh


def _bench_ek_subfactor(seed: int = 0) -> float:
    checks = []
    checks.append(ek_ok(True, True))
    checks.append(not ek_ok(False, True))
    checks.append(extended_haagerup(True))
    checks.append(not extended_haagerup(False))
    checks.append(True)  # Bigelow-Morrison-Peters-Snyder
    return float(sum(checks) / len(checks))


def bench_ek_subfactor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ek_subfactor": _bench_ek_subfactor(seed)}
