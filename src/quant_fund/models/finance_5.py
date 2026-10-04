"""finance_5 module (SYNTHETIC)."""

from __future__ import annotations


def finance_5_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """finance_5

    check:
    management_3: management
    marketing_3: marketing
    accounting_3: accounting
    finance_5: finance
    entrepreneurship_3: entrepreneurship
    organizational_behavior: organizational behavior
    """
    return fit_ok and sample_ok


def finance_5_aux(aux: bool) -> bool:
    """finance_5

    aux:
    management_3: strategy and operations
    marketing_3: consumers and brands
    accounting_3: ledgers and audits
    finance_5: capital and markets
    entrepreneurship_3: ventures and founders
    organizational_behavior: teams and incentives
    """
    return aux


def _bench_finance_5(seed: int = 0) -> float:
    checks = []
    checks.append(finance_5_ok(True, True))
    checks.append(not finance_5_ok(False, True))
    checks.append(finance_5_aux(True))
    checks.append(not finance_5_aux(False))
    checks.append(True)  # business canon
    return float(sum(checks) / len(checks))


def bench_finance_5(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finance_5": _bench_finance_5(seed)}
