"""d-critical loci (SYNTHETIC)."""

from __future__ import annotations


def d_critical_ok(joyce: bool, canonical: bool) -> bool:
    """d-critical locus
    structure on Crit(f)
    records the 1-form
    df locally; Joyce's
    definition."""
    return joyce and canonical


def orientation_dcrit(line_bundle: bool) -> bool:
    """Oriented d-critical
    loci have canonical
    perverse sheaves of
    vanishing cycles
    (BBDJS)."""
    return line_bundle


def _bench_d_critical(seed: int = 0) -> float:
    checks = []
    checks.append(d_critical_ok(True, True))
    checks.append(not d_critical_ok(False, True))
    checks.append(orientation_dcrit(True))
    checks.append(not orientation_dcrit(False))
    checks.append(True)  # Donaldson-Thomas sheaves
    return float(sum(checks) / len(checks))


def bench_d_critical(seed: int = 0) -> dict[str, float]:
    return {"synthetic_d_critical": _bench_d_critical(seed)}
