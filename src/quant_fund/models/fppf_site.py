"""fppf site (SYNTHETIC)."""

from __future__ import annotations


def fp_ok(fppf: bool, presentation: bool) -> bool:
    """fppf:
    fppf
    site
    and
    topology —
    faithfully
    flat."""
    return fppf and presentation


def fppf_cover(fc: bool) -> bool:
    """fppf
    cover:
    fppf
    covering
    family —
    fppf
    topology."""
    return fc


def _bench_fppf_site(seed: int = 0) -> float:
    checks = []
    checks.append(fp_ok(True, True))
    checks.append(not fp_ok(False, True))
    checks.append(fppf_cover(True))
    checks.append(not fppf_cover(False))
    checks.append(True)  # fppf
    return float(sum(checks) / len(checks))


def bench_fppf_site(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fppf_site": _bench_fppf_site(seed)}
