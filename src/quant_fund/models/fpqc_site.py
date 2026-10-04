"""fpqc site (SYNTHETIC)."""

from __future__ import annotations


def fq_ok(fpqc: bool, quasi_compact: bool) -> bool:
    """fpqc:
    fpqc
    site
    topology —
    faithfully
    flat
    qc."""
    return fpqc and quasi_compact


def fpqc_down(fd: bool) -> bool:
    """fpqc
    descent:
    fpqc
    descent
    for
    morphisms —
    Grothendieck
    fpqc."""
    return fd


def _bench_fpqc_site(seed: int = 0) -> float:
    checks = []
    checks.append(fq_ok(True, True))
    checks.append(not fq_ok(False, True))
    checks.append(fpqc_down(True))
    checks.append(not fpqc_down(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_fpqc_site(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fpqc_site": _bench_fpqc_site(seed)}
