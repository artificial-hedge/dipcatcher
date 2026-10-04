"""penology_2 module (SYNTHETIC)."""

from __future__ import annotations


def penology_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """penology_2

    check:
    criminology_3: criminology
    forensic_science_2: forensic science
    penology_2: penology
    victimology_2: victimology
    security_studies_2: security studies
    intelligence_studies_2: intelligence studies
    """
    return fit_ok and sample_ok


def penology_2_aux(aux: bool) -> bool:
    """penology_2

    aux:
    criminology_3: offenses and offenders
    forensic_science_2: traces and analysis
    penology_2: punishment and corrections
    victimology_2: victims and harm
    security_studies_2: threats and strategy
    intelligence_studies_2: collection and assessment
    """
    return aux


def _bench_penology_2(seed: int = 0) -> float:
    checks = []
    checks.append(penology_2_ok(True, True))
    checks.append(not penology_2_ok(False, True))
    checks.append(penology_2_aux(True))
    checks.append(not penology_2_aux(False))
    checks.append(True)  # justice canon
    return float(sum(checks) / len(checks))


def bench_penology_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_penology_2": _bench_penology_2(seed)}
