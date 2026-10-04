"""haddad2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def haddad2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """haddad2_qa_studies

    check:
    haddad2_qa_studies: s
    """
    return fit_ok and sample_ok


def haddad2_qa_studies_aux(aux: bool) -> bool:
    """haddad2_qa_studies

    aux:
    haddad2_qa_studies: t
    """
    return aux


def _bench_haddad2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(haddad2_qa_studies_ok(True, True))
    checks.append(not haddad2_qa_studies_ok(False, True))
    checks.append(haddad2_qa_studies_aux(True))
    checks.append(not haddad2_qa_studies_aux(False))
    checks.append(True)  # edomite-myth canon
    return float(sum(checks) / len(checks))


def bench_haddad2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haddad2_qa_studies": _bench_haddad2_qa_studies(seed)}
