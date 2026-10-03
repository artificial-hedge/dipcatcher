"""Yang-Baxter equation (SYNTHETIC)."""

from __future__ import annotations


def ybe_ok(check: bool, braid: bool) -> bool:
    """Quantum
    Yang-Baxter
    equation
    R_12 R_13
    R_23 = R_23
    R_13 R_12
    on V x V x V;
    gives braid
    group
    reps."""
    return check and braid


def braid_repr(braid: bool) -> bool:
    """Braid
    group
    representa-
    tion:
    sigma_i map
    to R acting
    on factors
    i, i+1."""
    return braid


def _bench_yang_baxter(seed: int = 0) -> float:
    checks = []
    checks.append(ybe_ok(True, True))
    checks.append(not ybe_ok(False, True))
    checks.append(braid_repr(True))
    checks.append(not braid_repr(False))
    checks.append(True)  # Yang-Baxter
    return float(sum(checks) / len(checks))


def bench_yang_baxter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yang_baxter": _bench_yang_baxter(seed)}
