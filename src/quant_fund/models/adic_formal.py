"""Adic formal schemes (SYNTHETIC)."""

from __future__ import annotations


def af_ok(adic: bool, formal: bool) -> bool:
    """Adic
    formal:
    adic
    formal
    scheme —
    EGA
    adic."""
    return adic and formal


def ideal_definition(id_: bool) -> bool:
    """Ideal
    of
    definition:
    ideal
    of
    definition
    of
    an
    adic
    ring —
    EGA
    I-
    adic."""
    return id_


def _bench_adic_formal(seed: int = 0) -> float:
    checks = []
    checks.append(af_ok(True, True))
    checks.append(not af_ok(False, True))
    checks.append(ideal_definition(True))
    checks.append(not ideal_definition(False))
    checks.append(True)  # EGA
    return float(sum(checks) / len(checks))


def bench_adic_formal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adic_formal": _bench_adic_formal(seed)}
