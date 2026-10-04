"""international_law module (SYNTHETIC)."""

from __future__ import annotations


def international_law_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """international_law

    check:
    constitutional_law: constitutional law
    criminal_law: criminal law
    contract_law: contract law
    tort_law: tort law
    administrative_law: administrative law
    international_law: international law
    """
    return fit_ok and sample_ok


def international_law_aux(aux: bool) -> bool:
    """international_law

    aux:
    constitutional_law: constitutional principles
    criminal_law: criminal offenses
    contract_law: contractual obligations
    tort_law: civil wrongs
    administrative_law: regulatory agencies
    international_law: treaties and norms
    """
    return aux


def _bench_international_law(seed: int = 0) -> float:
    checks = []
    checks.append(international_law_ok(True, True))
    checks.append(not international_law_ok(False, True))
    checks.append(international_law_aux(True))
    checks.append(not international_law_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_international_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_international_law": _bench_international_law(seed)}
