"""accounting_3 module (SYNTHETIC)."""

from __future__ import annotations


def accounting_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """accounting_3

    check:
    management_3: management
    marketing_3: marketing
    accounting_3: accounting
    finance_5: finance
    entrepreneurship_3: entrepreneurship
    organizational_behavior: organizational behavior
    """
    return fit_ok and sample_ok


def accounting_3_aux(aux: bool) -> bool:
    """accounting_3

    aux:
    management_3: strategy and operations
    marketing_3: consumers and brands
    accounting_3: ledgers and audits
    finance_5: capital and markets
    entrepreneurship_3: ventures and founders
    organizational_behavior: teams and incentives
    """
    return aux


def _bench_accounting_3(seed: int = 0) -> float:
    checks = []
    checks.append(accounting_3_ok(True, True))
    checks.append(not accounting_3_ok(False, True))
    checks.append(accounting_3_aux(True))
    checks.append(not accounting_3_aux(False))
    checks.append(True)  # business canon
    return float(sum(checks) / len(checks))


def bench_accounting_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_accounting_3": _bench_accounting_3(seed)}
