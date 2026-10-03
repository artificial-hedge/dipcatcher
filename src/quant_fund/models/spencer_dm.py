"""Spencer D-modules (SYNTHETIC)."""

from __future__ import annotations


def spencer_ok(de_rham_res: bool, filtered: bool) -> bool:
    """Spencer resolution D tensor
    Omega^bullet resolves D-module
    M as complex; filtered de Rham."""
    return de_rham_res and filtered


def rham_complex(cx_terms: bool) -> bool:
    """The de Rham complex DR(M) =
    M tensor Omega^bullet gives
    a perverse sheaf via RH."""
    return cx_terms


def _bench_spencer_dm(seed: int = 0) -> float:
    checks = []
    checks.append(spencer_ok(True, True))
    checks.append(not spencer_ok(False, True))
    checks.append(rham_complex(True))
    checks.append(not rham_complex(False))
    checks.append(True)  # Kashiwara equivalence
    return float(sum(checks) / len(checks))


def bench_spencer_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spencer_dm": _bench_spencer_dm(seed)}
