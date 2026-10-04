"""finance_4 module (SYNTHETIC)."""

from __future__ import annotations


def finance_4_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """finance_4

    check:
    accounting_2: accounting
    finance_4: finance
    marketing_2: marketing
    management_2: management
    entrepreneurship_2: entrepreneurship
    business_administration: business administration
    """
    return fit_ok and sample_ok


def finance_4_aux(aux: bool) -> bool:
    """finance_4

    aux:
    accounting_2: ledgers and audits
    finance_4: portfolios and valuations
    marketing_2: segments and campaigns
    management_2: teams and operations
    entrepreneurship_2: ventures and pivots
    business_administration: strategy and governance
    """
    return aux


def _bench_finance_4(seed: int = 0) -> float:
    checks = []
    checks.append(finance_4_ok(True, True))
    checks.append(not finance_4_ok(False, True))
    checks.append(finance_4_aux(True))
    checks.append(not finance_4_aux(False))
    checks.append(True)  # business canon
    return float(sum(checks) / len(checks))


def bench_finance_4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finance_4": _bench_finance_4(seed)}
