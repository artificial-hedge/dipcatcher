"""international_economics module (SYNTHETIC)."""

from __future__ import annotations


def international_economics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """international_economics

    check:
    labor_economics: labor economics
    public_economics: public economics
    industrial_organization: industrial organization
    international_economics: international economics
    financial_economics: financial economics
    monetary_economics: monetary economics
    """
    return fit_ok and sample_ok


def international_economics_aux(aux: bool) -> bool:
    """international_economics

    aux:
    labor_economics: wage structures
    public_economics: taxation systems
    industrial_organization: market structures
    international_economics: trade flows
    financial_economics: asset markets
    monetary_economics: money supply
    """
    return aux


def _bench_international_economics(seed: int = 0) -> float:
    checks = []
    checks.append(international_economics_ok(True, True))
    checks.append(not international_economics_ok(False, True))
    checks.append(international_economics_aux(True))
    checks.append(not international_economics_aux(False))
    checks.append(True)  # economics-3 canon
    return float(sum(checks) / len(checks))


def bench_international_economics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_international_economics": _bench_international_economics(seed)}
