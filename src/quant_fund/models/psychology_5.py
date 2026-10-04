"""psychology_5 module (SYNTHETIC)."""

from __future__ import annotations


def psychology_5_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """psychology_5

    check:
    sociology_6: sociology
    economics_6: economics
    political_science_3: political science
    psychology_5: psychology
    anthropology_6: anthropology
    linguistics_7: linguistics
    """
    return fit_ok and sample_ok


def psychology_5_aux(aux: bool) -> bool:
    """psychology_5

    aux:
    sociology_6: groups and institutions
    economics_6: markets and incentives
    political_science_3: power and governance
    psychology_5: minds and behavior
    anthropology_6: cultures and fieldwork
    linguistics_7: grammars and corpora
    """
    return aux


def _bench_psychology_5(seed: int = 0) -> float:
    checks = []
    checks.append(psychology_5_ok(True, True))
    checks.append(not psychology_5_ok(False, True))
    checks.append(psychology_5_aux(True))
    checks.append(not psychology_5_aux(False))
    checks.append(True)  # social-sciences canon
    return float(sum(checks) / len(checks))


def bench_psychology_5(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psychology_5": _bench_psychology_5(seed)}
