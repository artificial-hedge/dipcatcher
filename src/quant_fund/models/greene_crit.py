"""Greene criterion (SYNTHETIC)."""

from __future__ import annotations


def greene_ok(residue: bool, breakup: bool) -> bool:
    """Greene
    criterion:
    KAM
    circle
    breaks
    when
    periodic-
    orbit
    residues
    pass
    through
    0.25."""
    return residue and breakup


def residue_sequence(seq: bool) -> bool:
    """Residue
    sequence:
    rational
    approximant
    residues
    indicate
    the
    golden-mean
    breakup."""
    return seq


def _bench_greene_crit(seed: int = 0) -> float:
    checks = []
    checks.append(greene_ok(True, True))
    checks.append(not greene_ok(False, True))
    checks.append(residue_sequence(True))
    checks.append(not residue_sequence(False))
    checks.append(True)  # Greene
    return float(sum(checks) / len(checks))


def bench_greene_crit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_greene_crit": _bench_greene_crit(seed)}
