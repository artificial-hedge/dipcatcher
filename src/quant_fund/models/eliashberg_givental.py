"""Eliashberg-Givental-Hofer program (SYNTHETIC)."""

from __future__ import annotations


def eg_ok(program: bool, symplectic_invariant: bool) -> bool:
    """Eliashberg-
    Givental-
    Hofer:
    SFT
    program
    unifying
    contact
    and
    symplectic
    invariants —
    announced
    2000."""
    return program and symplectic_invariant


def rational_sft(rs: bool) -> bool:
    """Rational
    SFT:
    genus-
    zero
    specialization
    of
    the
    full
    SFT
    algebra —
    hierarchy
    level."""
    return rs


def _bench_eliashberg_givental(seed: int = 0) -> float:
    checks = []
    checks.append(eg_ok(True, True))
    checks.append(not eg_ok(False, True))
    checks.append(rational_sft(True))
    checks.append(not rational_sft(False))
    checks.append(True)  # EGH 2000
    return float(sum(checks) / len(checks))


def bench_eliashberg_givental(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eliashberg_givental": _bench_eliashberg_givental(seed)}
